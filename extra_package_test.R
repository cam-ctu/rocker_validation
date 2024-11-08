# make  a holding directory pack
pkg <- "ggalluvial"
# check if there is a test folder in the installed library.

dl <- download.packages(pkg, destdir="package",type="source",
                        method="wget")# wget was what solved this.. was getting binary versions

#unzip the tar.gz file
# 

unlink(paste0("package/",pkg))
untar(dl[,2], exdir = "package", restore_times=FALSE)

# modify the testInstalledpackages to use the /test fold from within pack.
test_pkg("ggalluvial", outDir="package/ggalluvial", test_path="tests")
# the test_path is relative to the outDir... c
# the above runs the test, the below does not as /test directory was not installed by default.
#tools::testInstalledPackage("ggalluvial", outDir="~/pack")

