from conans import ConanFile, CMake

class RegexConan(ConanFile):
    name = "ska-sdp-func"
    version = "1.1.7"
    settings = "os", "arch", "compiler", "build_type"
    generators = "cmake"

    def export_sources(self):
        self.copy("src/*", src="../..")                 # -> copies all .cpp files from working dir to a "source" dir
        self.copy("cmake/*", src="../..")               # -> copies cmake files from working dir to a "source" dir
        self.copy("tests/*", src="../..")               # -> copies tests files from working dir to a "source" dir
        self.copy("CMakeLists.txt", src="../..")        # -> copies CMakeLists.txt from working dir to a "source" dir

    def build(self):
        cmake = CMake(self)                # CMake helper auto-formats CLI arguments for CMake
        cmake.configure()                  # cmake -DCMAKE_TOOLCHAIN_FILE=conantoolchain.cmake
        cmake.build()                      # cmake --build .
        cmake.test()                       # cmake --build . --target=test

    def package(self):
        self.copy("MANIFEST.skao.int", src="src")
        cmake = CMake(self)                # For CMake projects which define an install target, leverage it
        cmake.install()                    # cmake --build . --target=install
                                           # sets CMAKE_INSTALL_PREFIX = <appropriate directory in conan cache>
        self.copy("*.h", dst="include", src="src")
        self.copy("*.a", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["SKASDPFuncPackage"]
