import os
import subprocess
from glob import glob

import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension

__version__ = "0.0.1"


# 用于设置编译选项
class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""

    def __str__(self):
        import pybind11
        return pybind11.get_include()


class CustomBuildExt(build_ext):
    def get_ext_filename(self, ext_name):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        # 设置编译后的扩展模块输出的文件名和位置
        ext_name = '{a}/{b}.so'.format(a=root_dir, b=ext_name)
        return ext_name


ext_modules = [Pybind11Extension('octree_module', sorted(glob('src/octree_module.cpp')),
                                 include_dirs=[
                                     get_pybind_include(),  # pybind11 头文件
                                 ],
                                 language='c++')
               ]

setup(
    name="octree_module",
    ext_modules=ext_modules,
    # package_dir={'utils.RPCC.ops': 'utils/RPCC/ops'},  # 指定包目录
    cmdclass={"build_ext": CustomBuildExt},
)
