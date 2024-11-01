# make  a holding directory pack
pkg <- "ggalluvial"
# check if there is a test folder in the installed library.

dl <- download.packages(pkg, destdir="~/pack",type="source",
                        method="wget")# wget was what solved this.. was getting binary versions

#unzip the tar.gz file
# 


file <- list.files("~/pack", full.names = TRUE)
untar(file, exdir = normalizePath("~/pack"))
# modify the testInstalledpackages to use the /test fold from within pack.
test_pkg("ggalluvial", outDir="~/pack", test_path=normalizePath("~/pack/ggalluvial/tests"))
# the above runs the test, the below does not as /test directory was not installed by default.
tools::testInstalledPackage("ggalluvial", outDir="~/pack")

