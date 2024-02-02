#Top level Rscript to do the validation with minimal input
# Make folder local, and stampe with docker version. Maybe alter the compose.yml to put folders in correct place.
#  Get Rversion, get docker details. 
# docker buildx imagetools inspect shug0131/cctu:4.3.1 --format "{{ json .Provenance.SLSA }}"
# docker manifest inspect --verbose shug0131/cctu:4.3.1
# Might want to make sure that the digest sha6:askjsad902130912,   match up
# But really that is no in the spirit of IP

#  Run a rmarkdown file to catch the output from running tools::test
#  give pointers to the docker setup, and the background links
# Explan the approach to packages and dates of repository.  Slides?

#  Then copy across all the testing outputs.
