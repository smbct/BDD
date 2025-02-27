import glob
import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from pkg_resources import parse_version

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""

    # from whichcraft import which
    from shutil import which

    return which(name) is not None

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = parse_version(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < parse_version('3.1.0'):
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def _validate_gcc_version(self, gcc_command):
        print(f'Testing {gcc_command}...')
        out = subprocess.check_output([gcc_command, '--version']).decode()
        if 'clang' in out.lower():
            return False
        words = out.split('\n')[0].split(' ')
        for word in reversed(words):
            if "." in word:
                gcc_version = parse_version(word)
                print(f"...has version {gcc_version}")
                if gcc_version >= parse_version('8.0'):
                    return True

        return False

    def _get_all_gcc_commands(self):
        all_path_dirs = subprocess.check_output("echo -n $PATH", shell=True).decode("utf-8").rstrip().split(":")

        all_gcc_commands = ['gcc']
        for path_dir in all_path_dirs:
            if not os.path.exists(path_dir):
                continue
            local_gccs = [s for s in os.listdir(path_dir) if re.search(r'^gcc-[0-9].?.?.?', s)]
            local_gccs = [s for s in local_gccs if os.access(os.path.join(path_dir, s), os.X_OK)]
            all_gcc_commands.extend(local_gccs)
        return all_gcc_commands


    def _find_suitable_gcc_gpp(self):
        # lists all gcc version in PATH
        all_gccs = self._get_all_gcc_commands()

        for gcc in all_gccs:
            if self._validate_gcc_version(gcc):
                matching_gpp = gcc.replace("cc", "++")
                print(f'Found suitable gcc/g++ version {gcc} {matching_gpp}')
                return gcc, matching_gpp

        raise RuntimeError("gcc >= 8.0 not found on the system")


    def _prepare_environment(self):
        gcc, gpp = self._find_suitable_gcc_gpp()

        gcc_path = subprocess.check_output(f"which {gcc}", shell=True).decode("utf-8").rstrip()
        gpp_path = subprocess.check_output(f"which {gpp}", shell=True).decode("utf-8").rstrip()

        os.environ["CC"] = gcc_path
        os.environ["CXX"] = gpp_path

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable
                    ]
        if _cuda_flag == "ON":
            cmake_args.append('-DWITH_CUDA=ON')

            cmake_args.append('-DCMAKE_PREFIX_PATH=/usr/local/cuda')
            cmake_args.append('-DCMAKE_CUDA_ARCHITECTURES=native')
        # if 'CUDA_PATH' in os.environ:
        #     cuda_path = os.environ['CUDA_HOME']
        #     print(f'Using CUDA from {cuda_path}')
        #     cmake_args.append('-DCUDAToolkit_ROOT=' + cuda_path)
        if is_tool('ninja'):
            #   '-DCMAKE_GENERATOR=Ninja' # Use Ninja instead of make for faster compilation.
            cmake_args.append('-DCMAKE_GENERATOR=Ninja')

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j8']

        self._prepare_environment()
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.', '--target', ext.name] + build_args, cwd=self.build_temp)


_cmake_modules = [CMakeExtension(name='bdd_mp_py'), 
                CMakeExtension(name='ILP_instance_py'),
                CMakeExtension(name='bdd_solver_py')]
_py_modules = []
_cuda_flag = os.environ.get("WITH_CUDA", "ON")
if _cuda_flag == "ON":
    _cmake_modules.append(CMakeExtension(name='bdd_cuda_learned_mma_py'))
    _py_modules.extend(['bdd_cuda_torch.bdd_cuda_torch', 'bdd_cuda_torch.bdd_torch_base', 'bdd_cuda_torch.bdd_torch_learned_mma'])

setup(
    name='BDD',
    version='0.0.3',
    description='Bindings for solving 0-1 integer linear programs via BDDs',
    packages=find_packages('src'),
    package_dir={'':'src'},
    ext_package='BDD',
    ext_modules=_cmake_modules,
    py_modules=_py_modules,
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    setup_requires=['wheel']
)
