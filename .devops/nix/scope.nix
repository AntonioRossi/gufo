{
  lib,
  newScope,
  releaseVersion ? null,
  revision,
  hy4Sources,
  rocmPackages,
  rustToolchain,
}:

lib.makeScope newScope (self: {
  # XRT with its amdxdna plugin: the XDNA NPU runtime (src/core/npu).
  xrt-amdxdna = (self.callPackage ./xrt { }).combined;
  # mlir-aie (IRON) and Peano for NPU designs (src/core/npu/kernels).
  iron = self.callPackage ./iron { };
  gufo = self.callPackage ./package.nix {
    inherit hy4Sources releaseVersion revision rocmPackages rustToolchain;
    xrt = self.xrt-amdxdna;
    inherit (self) iron;
  };
  mkServe = self.callPackage ./mk-serve.nix { };
  extensionModules = self.callPackage ./extension-modules.nix { };
})
