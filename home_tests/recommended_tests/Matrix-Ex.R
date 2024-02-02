pkgname <- "Matrix"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('Matrix')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("BunchKaufman-methods")
### * BunchKaufman-methods

flush(stderr()); flush(stdout())

### Name: BunchKaufman-methods
### Title: Bunch-Kaufman Decomposition Methods
### Aliases: BunchKaufman BunchKaufman-methods
###   BunchKaufman,dspMatrix-method BunchKaufman,dsyMatrix-method
###   BunchKaufman,matrix-method
### Keywords: methods

### ** Examples

data(CAex)
dim(CAex)
isSymmetric(CAex)# TRUE
CAs <- as(CAex, "symmetricMatrix")
if(FALSE) # no method defined yet for *sparse* :
   bk. <- BunchKaufman(CAs)
## does apply to *dense* symmetric matrices:
bkCA <- BunchKaufman(as(CAs, "denseMatrix"))
bkCA
pkCA <- pack(bkCA)
stopifnot(is(bkCA, "triangularMatrix"),
          is(pkCA, "triangularMatrix"),
          is(pkCA, "packedMatrix"))

image(bkCA)# shows how sparse it is, too
str(R.CA <- as(bkCA, "sparseMatrix"))
## an upper triangular 72x72 matrix with only 144 non-zero entries
stopifnot(is(R.CA, "triangularMatrix"), is(R.CA, "CsparseMatrix"))



cleanEx()
nameEx("CAex")
### * CAex

flush(stderr()); flush(stdout())

### Name: CAex
### Title: Albers' example Matrix with "Difficult" Eigen Factorization
### Aliases: CAex
### Keywords: datasets

### ** Examples

data(CAex)
str(CAex) # of class "dgCMatrix"

image(CAex)# -> it's a simple band matrix with 5 bands
## and the eigen values are basically 1 (42 times) and 0 (30 x):
zapsmall(ev <- eigen(CAex, only.values=TRUE)$values)
## i.e., the matrix is symmetric, hence
sCA <- as(CAex, "symmetricMatrix")
## and
stopifnot(class(sCA) == "dsCMatrix",
          as(sCA, "matrix") == as(CAex, "matrix"))



cleanEx()
nameEx("CHMfactor-class")
### * CHMfactor-class

flush(stderr()); flush(stdout())

### Name: CHMfactor-class
### Title: CHOLMOD-based Cholesky Factorizations
### Aliases: CHMfactor-class CHMsimpl-class CHMsuper-class dCHMsimpl-class
###   dCHMsuper-class nCHMsimpl-class nCHMsuper-class
###   coerce,CHMfactor,CsparseMatrix-method coerce,CHMfactor,Matrix-method
###   coerce,CHMfactor,RsparseMatrix-method
###   coerce,CHMfactor,TsparseMatrix-method coerce,CHMfactor,dMatrix-method
###   coerce,CHMfactor,dsparseMatrix-method coerce,CHMfactor,pMatrix-method
###   coerce,CHMfactor,sparseMatrix-method
###   coerce,CHMfactor,triangularMatrix-method
###   determinant,CHMfactor,logical-method update,CHMfactor-method
###   .updateCHMfactor isLDL
### Keywords: classes algebra

### ** Examples

## An example for the expand() method
n <- 1000; m <- 200; nnz <- 2000
set.seed(1)
M1 <- spMatrix(n, m,
               i = sample(n, nnz, replace = TRUE),
               j = sample(m, nnz, replace = TRUE),
               x = round(rnorm(nnz),1))
XX <- crossprod(M1) ## = M1'M1  = M M'  where M <- t(M1)
CX <- Cholesky(XX)
isLDL(CX)
str(CX) ## a "dCHMsimpl" object
r <- expand(CX)
L.P <- with(r, crossprod(L,P))  ## == L'P
PLLP <- crossprod(L.P)          ## == (L'P)' L'P == P'LL'P  = XX = M M'
b <- sample(m)
stopifnot(all.equal(PLLP, XX), 
          all(as.vector(solve(CX, b, system="P" )) == r$P %*% b),
          all(as.vector(solve(CX, b, system="Pt")) == t(r$P) %*% b) )

u1 <- update(CX, XX,    mult=pi)
u2 <- update(CX, t(M1), mult=pi) # with the original M, where XX = M M'
stopifnot(all.equal(u1,u2, tol=1e-14))

   ## [ See  help(Cholesky)  for more examples ]
   ##        -------------



cleanEx()
nameEx("Cholesky-class")
### * Cholesky-class

flush(stderr()); flush(stdout())

### Name: Cholesky-class
### Title: Cholesky and Bunch-Kaufman Decompositions
### Aliases: Cholesky-class pCholesky-class BunchKaufman-class
###   pBunchKaufman-class show,BunchKaufman-method
###   show,pBunchKaufman-method show,Cholesky-method show,pCholesky-method
### Keywords: classes algebra

### ** Examples

(sm <- pack(Matrix(diag(5) + 1))) # dspMatrix
signif(csm <- chol(sm), 4)

(pm <- crossprod(Matrix(rnorm(18), nrow = 6, ncol = 3)))
(ch <- chol(pm))
if (toupper(ch@uplo) == "U") # which is TRUE
   crossprod(ch)
stopifnot(all.equal(as(crossprod(ch), "matrix"),
                    as(pm, "matrix"), tolerance=1e-14))



cleanEx()
nameEx("Cholesky")
### * Cholesky

flush(stderr()); flush(stdout())

### Name: Cholesky
### Title: Cholesky Decomposition of a Sparse Matrix
### Aliases: Cholesky Cholesky,denseMatrix-method Cholesky,dsCMatrix-method
###   Cholesky,nsparseMatrix-method Cholesky,sparseMatrix-method
###   .SuiteSparse_version
### Keywords: array algebra

### ** Examples

data(KNex)
mtm <- with(KNex, crossprod(mm))
str(mtm@factors) # empty list()
(C1 <- Cholesky(mtm))             # uses show(<MatrixFactorization>)
str(mtm@factors) # 'sPDCholesky' (simpl)
(Cm <- Cholesky(mtm, super = TRUE))
c(C1 = isLDL(C1), Cm = isLDL(Cm))
str(mtm@factors) # 'sPDCholesky'  *and* 'SPdCholesky'
str(cm1  <- as(C1, "sparseMatrix"))
str(cmat <- as(Cm, "sparseMatrix"))# hmm: super is *less* sparse here
cm1[1:20, 1:20]

b <- matrix(c(rep(0, 711), 1), ncol = 1)
## solve(Cm, b) by default solves  Ax = b, where A = Cm'Cm (= mtm)!
## hence, the identical() check *should* work, but fails on some GOTOblas:
x <- solve(Cm, b)
stopifnot(identical(x, solve(Cm, b, system = "A")),
          all.equal(x, solve(mtm, b)))

Cn <- Cholesky(mtm, perm = FALSE)# no permutation -- much worse:
sizes <- c(simple = object.size(C1),
           super  = object.size(Cm),
           noPerm = object.size(Cn))
## simple is 100, super= 137, noPerm= 812 :
noquote(cbind(format(100 * sizes / sizes[1], digits=4)))


## Visualize the sparseness:
dq <- function(ch) paste('"',ch,'"', sep="") ## dQuote(<UTF-8>) gives bad plots
image(mtm, main=paste("crossprod(mm) : Sparse", dq(class(mtm))))
image(cm1, main= paste("as(Cholesky(crossprod(mm)),\"sparseMatrix\"):",
                        dq(class(cm1))))
## Don't show: 
expand(C1) ## to check printing
## End(Don't show)

## Smaller example, with same matrix as in  help(chol) :
(mm <- Matrix(toeplitz(c(10, 0, 1, 0, 3)), sparse = TRUE)) # 5 x 5
(opts <- expand.grid(perm = c(TRUE,FALSE), LDL = c(TRUE,FALSE), super = c(FALSE,TRUE)))
rr <- lapply(seq_len(nrow(opts)), function(i)
             do.call(Cholesky, c(list(A = mm), opts[i,])))
nn <- do.call(expand.grid, c(attr(opts, "out.attrs")$dimnames,
              stringsAsFactors=FALSE,KEEP.OUT.ATTRS=FALSE))
names(rr) <- apply(nn, 1, function(r)
                   paste(sub("(=.).*","\\1", r), collapse=","))
str(rr, max.level=1)

str(re <- lapply(rr, expand), max.level=2) ## each has a 'P' and a 'L' matrix
R0 <- chol(mm, pivot=FALSE)
R1 <- chol(mm, pivot=TRUE )
stopifnot(all.equal(t(R1), re[[1]]$L),
          all.equal(t(R0), re[[2]]$L),
          identical(as(1:5, "pMatrix"), re[[2]]$P), # no pivoting
TRUE)

## Don't show: 
str(dd <- .diag.dsC(mtm))
dc <- .diag.dsC(Chx=C1) # <- directly from the Cholesky
stopifnot(all.equal(dd,dc))
## End(Don't show)
# Version of the underlying SuiteSparse library by Tim Davis :
.SuiteSparse_version()



cleanEx()
nameEx("CsparseMatrix-class")
### * CsparseMatrix-class

flush(stderr()); flush(stdout())

### Name: CsparseMatrix-class
### Title: Class "CsparseMatrix" of Sparse Matrices in Column-compressed
###   Form
### Aliases: CsparseMatrix-class Arith,CsparseMatrix,CsparseMatrix-method
###   Arith,CsparseMatrix,numeric-method Arith,numeric,CsparseMatrix-method
###   Compare,CsparseMatrix,CsparseMatrix-method
###   Logic,CsparseMatrix,CsparseMatrix-method Math,CsparseMatrix-method
###   as.vector,CsparseMatrix-method
###   coerce,CsparseMatrix,RsparseMatrix-method
###   coerce,CsparseMatrix,TsparseMatrix-method
###   coerce,CsparseMatrix,denseMatrix-method
###   coerce,CsparseMatrix,generalMatrix-method
###   coerce,CsparseMatrix,matrix-method
###   coerce,CsparseMatrix,packedMatrix-method
###   coerce,CsparseMatrix,sparseVector-method
###   coerce,CsparseMatrix,unpackedMatrix-method
###   coerce,CsparseMatrix,vector-method coerce,matrix,CsparseMatrix-method
###   coerce,numLike,CsparseMatrix-method diag,CsparseMatrix-method
###   diag<-,CsparseMatrix-method log,CsparseMatrix-method
###   t,CsparseMatrix-method .validateCsparse
### Keywords: classes

### ** Examples

getClass("CsparseMatrix")

## The common validity check function (based on C code):
getValidity(getClass("CsparseMatrix"))



cleanEx()
nameEx("Diagonal")
### * Diagonal

flush(stderr()); flush(stdout())

### Name: Diagonal
### Title: Construct a Diagonal Matrix
### Aliases: Diagonal .sparseDiagonal .trDiagonal .symDiagonal
### Keywords: array algebra

### ** Examples

Diagonal(3)
Diagonal(x = 10^(3:1))
Diagonal(x = (1:4) >= 2)#-> "ldiMatrix"

## Use Diagonal() + kronecker() for "repeated-block" matrices:
M1 <- Matrix(0+0:5, 2,3)
(M <- kronecker(Diagonal(3), M1))

(S <- crossprod(Matrix(rbinom(60, size=1, prob=0.1), 10,6)))
(SI <- S + 10*.symDiagonal(6)) # sparse symmetric still
stopifnot(is(SI, "dsCMatrix"))
(I4 <- .sparseDiagonal(4, shape="t"))# now (2012-10) unitriangular
stopifnot(I4@diag == "U", all(I4 == diag(4)))
## Don't show: 
  L <- Diagonal(5, TRUE)
  stopifnot(L@diag == "U", identical(L, Diagonal(5) > 0))
## End(Don't show)



cleanEx()
nameEx("Hilbert")
### * Hilbert

flush(stderr()); flush(stdout())

### Name: Hilbert
### Title: Generate a Hilbert matrix
### Aliases: Hilbert
### Keywords: array algebra

### ** Examples

Hilbert(6)



cleanEx()
nameEx("KNex")
### * KNex

flush(stderr()); flush(stdout())

### Name: KNex
### Title: Koenker-Ng Example Sparse Model Matrix and Response Vector
### Aliases: KNex
### Keywords: datasets

### ** Examples

data(KNex)
class(KNex$mm)
  dim(KNex$mm)
image(KNex$mm)
str(KNex)

system.time( # a fraction of a second
  sparse.sol <- with(KNex, solve(crossprod(mm), crossprod(mm, y))))

head(round(sparse.sol,3))

## Compare with QR-based solution ("more accurate, but slightly slower"):
system.time(
  sp.sol2 <- with(KNex, qr.coef(qr(mm), y) ))

all.equal(sparse.sol, sp.sol2, tolerance = 1e-13) # TRUE



cleanEx()
nameEx("KhatriRao")
### * KhatriRao

flush(stderr()); flush(stdout())

### Name: KhatriRao
### Title: Khatri-Rao Matrix Product
### Aliases: KhatriRao
### Keywords: methods array

### ** Examples

## Example with very small matrices:
m <- matrix(1:12,3,4)
d <- diag(1:4)
KhatriRao(m,d)
KhatriRao(d,m)
dimnames(m) <- list(LETTERS[1:3], letters[1:4])
KhatriRao(m,d, make.dimnames=TRUE)
KhatriRao(d,m, make.dimnames=TRUE)
dimnames(d) <- list(NULL, paste0("D", 1:4))
KhatriRao(m,d, make.dimnames=TRUE)
KhatriRao(d,m, make.dimnames=TRUE)
dimnames(d) <- list(paste0("d", 10*1:4), paste0("D", 1:4))
(Kmd <- KhatriRao(m,d, make.dimnames=TRUE))
(Kdm <- KhatriRao(d,m, make.dimnames=TRUE))

nm <- as(m, "nsparseMatrix")
nd <- as(d, "nsparseMatrix")
KhatriRao(nm,nd, make.dimnames=TRUE)
KhatriRao(nd,nm, make.dimnames=TRUE)

stopifnot(dim(KhatriRao(m,d)) == c(nrow(m)*nrow(d), ncol(d)))
## border cases / checks:
zm <- nm; zm[] <- FALSE # all FALSE matrix
stopifnot(all(K1 <- KhatriRao(nd, zm) == 0), identical(dim(K1), c(12L, 4L)),
          all(K2 <- KhatriRao(zm, nd) == 0), identical(dim(K2), c(12L, 4L)))

d0 <- d; d0[] <- 0; m0 <- Matrix(d0[-1,])
stopifnot(all(K3 <- KhatriRao(d0, m) == 0), identical(dim(K3), dim(Kdm)),
	  all(K4 <- KhatriRao(m, d0) == 0), identical(dim(K4), dim(Kmd)),
	  all(KhatriRao(d0, d0) == 0), all(KhatriRao(m0, d0) == 0),
	  all(KhatriRao(d0, m0) == 0), all(KhatriRao(m0, m0) == 0),
	  identical(dimnames(KhatriRao(m, d0, make.dimnames=TRUE)), dimnames(Kmd)))

## a matrix with "structural" and non-structural zeros:
m01 <- new("dgCMatrix", i = c(0L, 2L, 0L, 1L), p = c(0L, 0L, 0L, 2L, 4L),
           Dim = 3:4, x = c(1, 0, 1, 0))
D4 <- Diagonal(4, x=1:4) # "as" d
DU <- Diagonal(4)# unit-diagonal: uplo="U"
(K5  <- KhatriRao( d, m01))
K5d  <- KhatriRao( d, m01, sparseY=FALSE)
K5Dd <- KhatriRao(D4, m01, sparseY=FALSE)
K5Ud <- KhatriRao(DU, m01, sparseY=FALSE)
(K6  <- KhatriRao(diag(3),     t(m01)))
K6D  <- KhatriRao(Diagonal(3), t(m01))
K6d  <- KhatriRao(diag(3),     t(m01), sparseY=FALSE)
K6Dd <- KhatriRao(Diagonal(3), t(m01), sparseY=FALSE)
stopifnot(exprs = {
    all(K5 == K5d)
    identical(cbind(c(7L, 10L), c(3L, 4L)),
              which(K5 != 0, arr.ind = TRUE, useNames=FALSE))
    identical(K5d, K5Dd)
    identical(K6, K6D)
    all(K6 == K6d)
    identical(cbind(3:4, 1L),
              which(K6 != 0, arr.ind = TRUE, useNames=FALSE))
    identical(K6d, K6Dd)
})



cleanEx()
nameEx("LU-class")
### * LU-class

flush(stderr()); flush(stdout())

### Name: LU-class
### Title: LU (dense) Matrix Decompositions
### Aliases: LU-class denseLU-class
### Keywords: classes algebra

### ** Examples

set.seed(1)
mm <- Matrix(round(rnorm(9),2), nrow = 3)
mm
str(lum <- lu(mm))
elu <- expand(lum)
elu # three components: "L", "U", and "P", the permutation
elu$L %*% elu$U
(m2 <- with(elu, P %*% L %*% U)) # the same as 'mm'
stopifnot(all.equal(as(mm, "matrix"),
                    as(m2, "matrix")))



cleanEx()
nameEx("Matrix-class")
### * Matrix-class

flush(stderr()); flush(stdout())

### Name: Matrix-class
### Title: Virtual Class "Matrix" Class of Matrices
### Aliases: Matrix-class !,Matrix-method &,Matrix,ddiMatrix-method
###   &,Matrix,ldiMatrix-method *,Matrix,ddiMatrix-method
###   *,Matrix,ldiMatrix-method +,Matrix,missing-method
###   -,Matrix,missing-method Arith,Matrix,Matrix-method
###   Arith,Matrix,lsparseMatrix-method Arith,Matrix,nsparseMatrix-method
###   Logic,ANY,Matrix-method Logic,Matrix,ANY-method
###   Logic,Matrix,nMatrix-method Math2,Matrix-method Ops,ANY,Matrix-method
###   Ops,Matrix,ANY-method Ops,Matrix,NULL-method
###   Ops,Matrix,ddiMatrix-method Ops,Matrix,ldiMatrix-method
###   Ops,Matrix,matrix-method Ops,Matrix,sparseVector-method
###   Ops,NULL,Matrix-method Ops,matrix,Matrix-method Summary,Matrix-method
###   ^,Matrix,ddiMatrix-method ^,Matrix,ldiMatrix-method
###   as.array,Matrix-method as.logical,Matrix-method
###   as.matrix,Matrix-method as.numeric,Matrix-method
###   as.vector,Matrix-method cbind2,ANY,Matrix-method
###   cbind2,Matrix,ANY-method cbind2,Matrix,Matrix-method
###   cbind2,Matrix,NULL-method cbind2,Matrix,atomicVector-method
###   cbind2,Matrix,missing-method cbind2,NULL,Matrix-method
###   coerce,ANY,Matrix-method coerce,Matrix,complex-method
###   coerce,Matrix,corMatrix-method coerce,Matrix,diagonalMatrix-method
###   coerce,Matrix,dpoMatrix-method coerce,Matrix,dppMatrix-method
###   coerce,Matrix,indMatrix-method coerce,Matrix,integer-method
###   coerce,Matrix,logical-method coerce,Matrix,matrix-method
###   coerce,Matrix,numeric-method coerce,Matrix,pMatrix-method
###   coerce,Matrix,symmetricMatrix-method
###   coerce,Matrix,triangularMatrix-method coerce,Matrix,vector-method
###   coerce,matrix,Matrix-method cov2cor,Matrix-method
###   determinant,Matrix,missing-method determinant,Matrix,logical-method
###   diff,Matrix-method dim,Matrix-method dimnames,Matrix-method
###   dimnames<-,Matrix,NULL-method dimnames<-,Matrix,list-method
###   drop,Matrix-method eigen,Matrix,ANY,logical-method
###   eigen,Matrix,ANY,missing-method head,Matrix-method
###   initialize,Matrix-method length,Matrix-method mean,Matrix-method
###   rbind2,ANY,Matrix-method rbind2,Matrix,ANY-method
###   rbind2,Matrix,Matrix-method rbind2,Matrix,NULL-method
###   rbind2,Matrix,atomicVector-method rbind2,Matrix,missing-method
###   rbind2,NULL,Matrix-method rep,Matrix-method tail,Matrix-method
###   unname,Matrix,missing-method svd,Matrix-method det print.Matrix
### Keywords: classes algebra

### ** Examples

slotNames("Matrix")

cl <- getClass("Matrix")
names(cl@subclasses) # more than 40 ..

showClass("Matrix")#> output with slots and all subclasses

(M <- Matrix(c(0,1,0,0), 6, 4))
dim(M)
diag(M)
cm <- M[1:4,] + 10*Diagonal(4)
diff(M)
## can reshape it even :
dim(M) <- c(2, 12)
M
stopifnot(identical(M, Matrix(c(0,1,0,0), 2,12)),
          all.equal(det(cm),
                    determinant(as(cm,"matrix"), log=FALSE)$modulus,
                    check.attributes=FALSE))



cleanEx()
nameEx("Matrix")
### * Matrix

flush(stderr()); flush(stdout())

### Name: Matrix
### Title: Construct a Classed Matrix
### Aliases: Matrix
### Keywords: array algebra

### ** Examples

Matrix(0, 3, 2)             # 3 by 2 matrix of zeros -> sparse
Matrix(0, 3, 2, sparse=FALSE)# -> 'dense'

## 4 cases - 3 different results :
Matrix(0, 2, 2)              # diagonal !
Matrix(0, 2, 2, sparse=FALSE)# (ditto)
Matrix(0, 2, 2,               doDiag=FALSE)# -> sparse symm. "dsCMatrix"
Matrix(0, 2, 2, sparse=FALSE, doDiag=FALSE)# -> dense  symm. "dsyMatrix"

Matrix(1:6, 3, 2)           # a 3 by 2 matrix (+ integer warning)
Matrix(1:6 + 1, nrow=3)

## logical ones:
Matrix(diag(4) >  0) # -> "ldiMatrix" with diag = "U"
Matrix(diag(4) >  0, sparse=TRUE) #  (ditto)
Matrix(diag(4) >= 0) # -> "lsyMatrix" (of all 'TRUE')
## triangular
l3 <- upper.tri(matrix(,3,3))
(M <- Matrix(l3))   # -> "ltCMatrix"
Matrix(! l3)        # -> "ltrMatrix"
as(l3, "CsparseMatrix")# "lgCMatrix"

Matrix(1:9, nrow=3,
       dimnames = list(c("a", "b", "c"), c("A", "B", "C")))
(I3 <- Matrix(diag(3)))# identity, i.e., unit "diagonalMatrix"
str(I3) # note  'diag = "U"' and the empty 'x' slot

(A <- cbind(a=c(2,1), b=1:2))# symmetric *apart* from dimnames
Matrix(A)                    # hence 'dgeMatrix'
(As <- Matrix(A, dimnames = list(NULL,NULL)))# -> symmetric
forceSymmetric(A) # also symmetric, w/ symm. dimnames
stopifnot(is(As, "symmetricMatrix"),
          is(Matrix(0, 3,3), "sparseMatrix"),
          is(Matrix(FALSE, 1,1), "sparseMatrix"))



cleanEx()
nameEx("MatrixClass")
### * MatrixClass

flush(stderr()); flush(stdout())

### Name: MatrixClass
### Title: The Matrix (Super-) Class of a Class
### Aliases: MatrixClass
### Keywords: classes

### ** Examples

mkA <- setClass("A", contains="dgCMatrix")
(A <- mkA())
stopifnot(identical(
     MatrixClass("A"),
     "dgCMatrix"))



cleanEx()
nameEx("MatrixFactorization-class")
### * MatrixFactorization-class

flush(stderr()); flush(stdout())

### Name: MatrixFactorization-class
### Title: Class "MatrixFactorization" of Matrix Factorizations
### Aliases: MatrixFactorization-class CholeskyFactorization-class
###   determinant,MatrixFactorization,missing-method
###   dim,MatrixFactorization-method show,MatrixFactorization-method
### Keywords: classes

### ** Examples

showClass("MatrixFactorization")
getClass("CholeskyFactorization")



cleanEx()
nameEx("RsparseMatrix-class")
### * RsparseMatrix-class

flush(stderr()); flush(stdout())

### Name: RsparseMatrix-class
### Title: Class "RsparseMatrix" of Sparse Matrices in Row-compressed Form
### Aliases: RsparseMatrix-class as.vector,RsparseMatrix-method
###   coerce,RsparseMatrix,CsparseMatrix-method
###   coerce,RsparseMatrix,TsparseMatrix-method
###   coerce,RsparseMatrix,denseMatrix-method
###   coerce,RsparseMatrix,generalMatrix-method
###   coerce,RsparseMatrix,matrix-method
###   coerce,RsparseMatrix,packedMatrix-method
###   coerce,RsparseMatrix,unpackedMatrix-method
###   coerce,RsparseMatrix,vector-method coerce,matrix,RsparseMatrix-method
###   coerce,numLike,RsparseMatrix-method diag,RsparseMatrix-method
###   diag<-,RsparseMatrix-method t,RsparseMatrix-method
### Keywords: classes

### ** Examples

showClass("RsparseMatrix")



cleanEx()
nameEx("Schur-class")
### * Schur-class

flush(stderr()); flush(stdout())

### Name: Schur-class
### Title: Class "Schur" of Schur Matrix Factorizations
### Aliases: Schur-class
### Keywords: classes

### ** Examples

showClass("Schur")
Schur(M <- Matrix(c(1:7, 10:2), 4,4))
## Trivial, of course:
str(Schur(Diagonal(5)))

## for more examples, see Schur()



cleanEx()
nameEx("Schur")
### * Schur

flush(stderr()); flush(stdout())

### Name: Schur
### Title: Schur Decomposition of a Matrix
### Aliases: Schur Schur,Matrix,missing-method Schur,matrix,missing-method
###   Schur,dgeMatrix,logical-method Schur,diagonalMatrix,logical-method
###   Schur,dsyMatrix,logical-method Schur,generalMatrix,logical-method
###   Schur,matrix,logical-method Schur,symmetricMatrix,logical-method
###   Schur,triangularMatrix,logical-method
### Keywords: algebra

### ** Examples

Schur(Hilbert(9))              # Schur factorization (real eigenvalues)

(A <- Matrix(round(rnorm(5*5, sd = 100)), nrow = 5))
(Sch.A <- Schur(A))

eTA <- eigen(Sch.A@T)
str(SchA <- Schur(A, vectors=FALSE))# no 'T' ==> simple list
stopifnot(all.equal(eTA$values, eigen(A)$values, tolerance = 1e-13),
          all.equal(eTA$values,
                    local({z <- Sch.A@EValues
                           z[order(Mod(z), decreasing=TRUE)]}), tolerance = 1e-13),
          identical(SchA$T, Sch.A@T),
          identical(SchA$EValues, Sch.A@EValues))

## For the faint of heart, we provide Schur() also for traditional matrices:

a.m <- function(M) unname(as(M, "matrix"))
a <- a.m(A)
Sch.a <- Schur(a)
stopifnot(identical(Sch.a, list(Q = a.m(Sch.A @ Q),
				T = a.m(Sch.A @ T),
				EValues = Sch.A@EValues)),
	  all.equal(a, with(Sch.a, Q %*% T %*% t(Q)))
)



cleanEx()
nameEx("Subassign-methods")
### * Subassign-methods

flush(stderr()); flush(stdout())

### Name: [<--methods
### Title: Methods for "[<-" - Assigning to Subsets for 'Matrix'
### Aliases: [<--methods Subassign-methods
###   [<-,CsparseMatrix,Matrix,missing,replValue-method
###   [<-,CsparseMatrix,index,index,replValue-method
###   [<-,CsparseMatrix,index,index,sparseVector-method
###   [<-,CsparseMatrix,index,missing,replValue-method
###   [<-,CsparseMatrix,index,missing,sparseVector-method
###   [<-,CsparseMatrix,matrix,missing,replValue-method
###   [<-,CsparseMatrix,missing,index,replValue-method
###   [<-,CsparseMatrix,missing,index,sparseVector-method
###   [<-,Matrix,ANY,ANY,ANY-method [<-,Matrix,ANY,ANY,Matrix-method
###   [<-,Matrix,ANY,ANY,matrix-method [<-,Matrix,ANY,missing,Matrix-method
###   [<-,Matrix,ANY,missing,matrix-method
###   [<-,Matrix,ldenseMatrix,missing,replValue-method
###   [<-,Matrix,lsparseMatrix,missing,replValue-method
###   [<-,Matrix,matrix,missing,replValue-method
###   [<-,Matrix,missing,ANY,Matrix-method
###   [<-,Matrix,missing,ANY,matrix-method
###   [<-,Matrix,ndenseMatrix,missing,replValue-method
###   [<-,Matrix,nsparseMatrix,missing,replValue-method
###   [<-,RsparseMatrix,index,index,replValue-method
###   [<-,RsparseMatrix,index,index,sparseVector-method
###   [<-,RsparseMatrix,index,missing,replValue-method
###   [<-,RsparseMatrix,index,missing,sparseVector-method
###   [<-,RsparseMatrix,matrix,missing,replValue-method
###   [<-,RsparseMatrix,missing,index,replValue-method
###   [<-,RsparseMatrix,missing,index,sparseVector-method
###   [<-,TsparseMatrix,Matrix,missing,replValue-method
###   [<-,TsparseMatrix,index,index,replValue-method
###   [<-,TsparseMatrix,index,index,sparseVector-method
###   [<-,TsparseMatrix,index,missing,replValue-method
###   [<-,TsparseMatrix,index,missing,sparseVector-method
###   [<-,TsparseMatrix,matrix,missing,replValue-method
###   [<-,TsparseMatrix,missing,index,replValue-method
###   [<-,TsparseMatrix,missing,index,sparseVector-method
###   [<-,denseMatrix,index,index,replValue-method
###   [<-,denseMatrix,index,missing,replValue-method
###   [<-,denseMatrix,matrix,missing,replValue-method
###   [<-,denseMatrix,missing,index,replValue-method
###   [<-,denseMatrix,missing,missing,ANY-method
###   [<-,diagonalMatrix,index,index,replValue-method
###   [<-,diagonalMatrix,index,index,sparseMatrix-method
###   [<-,diagonalMatrix,index,index,sparseVector-method
###   [<-,diagonalMatrix,index,missing,replValue-method
###   [<-,diagonalMatrix,index,missing,sparseMatrix-method
###   [<-,diagonalMatrix,index,missing,sparseVector-method
###   [<-,diagonalMatrix,matrix,missing,replValue-method
###   [<-,diagonalMatrix,missing,index,replValue-method
###   [<-,diagonalMatrix,missing,index,sparseMatrix-method
###   [<-,diagonalMatrix,missing,index,sparseVector-method
###   [<-,diagonalMatrix,missing,missing,ANY-method
###   [<-,indMatrix,index,index,ANY-method
###   [<-,indMatrix,index,missing,ANY-method
###   [<-,indMatrix,missing,index,ANY-method
###   [<-,indMatrix,missing,missing,ANY-method
###   [<-,sparseMatrix,ANY,ANY,sparseMatrix-method
###   [<-,sparseMatrix,ANY,missing,sparseMatrix-method
###   [<-,sparseMatrix,missing,ANY,sparseMatrix-method
###   [<-,sparseMatrix,missing,missing,ANY-method
###   [<-,sparseVector,index,missing,replValueSp-method
###   [<-,sparseVector,sparseVector,missing,replValueSp-method
### Keywords: methods array

### ** Examples

set.seed(101)
(a <- m <- Matrix(round(rnorm(7*4),2), nrow = 7))

a[] <- 2.2 # <<- replaces **every** entry
a
## as do these:
a[,] <- 3 ; a[TRUE,] <- 4

m[2, 3]  <- 3.14 # simple number
m[3, 3:4]<- 3:4  # simple numeric of length 2

## sub matrix assignment:
m[-(4:7), 3:4] <- cbind(1,2:4) #-> upper right corner of 'm'
m[3:5, 2:3] <- 0
m[6:7, 1:2] <- Diagonal(2)
m

## rows or columns only:
m[1,] <- 10
m[,2] <- 1:7
m[-(1:6), ] <- 3:0 # not the first 6 rows, i.e. only the 7th
as(m, "sparseMatrix")



cleanEx()
nameEx("TsparseMatrix-class")
### * TsparseMatrix-class

flush(stderr()); flush(stdout())

### Name: TsparseMatrix-class
### Title: Class "TsparseMatrix" of Sparse Matrices in Triplet Form
### Aliases: TsparseMatrix-class as.vector,TsparseMatrix-method
###   coerce,TsparseMatrix,CsparseMatrix-method
###   coerce,TsparseMatrix,RsparseMatrix-method
###   coerce,TsparseMatrix,denseMatrix-method
###   coerce,TsparseMatrix,generalMatrix-method
###   coerce,TsparseMatrix,matrix-method
###   coerce,TsparseMatrix,packedMatrix-method
###   coerce,TsparseMatrix,sparseVector-method
###   coerce,TsparseMatrix,unpackedMatrix-method
###   coerce,TsparseMatrix,vector-method coerce,matrix,TsparseMatrix-method
###   coerce,numLike,TsparseMatrix-method diag,TsparseMatrix-method
###   diag<-,TsparseMatrix-method t,TsparseMatrix-method
### Keywords: classes

### ** Examples

showClass("TsparseMatrix")
## or just the subclasses' names
names(getClass("TsparseMatrix")@subclasses)

T3 <- spMatrix(3,4, i=c(1,3:1), j=c(2,4:2), x=1:4)
T3 # only 3 non-zero entries, 5 = 1+4 !
## Don't show: 
stopifnot(nnzero(T3) == 3)
## End(Don't show)



cleanEx()
nameEx("USCounties")
### * USCounties

flush(stderr()); flush(stdout())

### Name: USCounties
### Title: USCounties Contiguity Matrix
### Aliases: USCounties
### Keywords: datasets

### ** Examples

data(USCounties)
(n <- ncol(USCounties))
IM <- .symDiagonal(n)
nn <- 50
set.seed(1)
rho <- runif(nn, 0, 1)
system.time(MJ <- sapply(rho, function(x)
	determinant(IM - x * USCounties, logarithm = TRUE)$modulus))

## can be done faster, by update()ing the Cholesky factor:
nWC <- -USCounties
C1 <- Cholesky(nWC, Imult = 2)
system.time(MJ1 <- n * log(rho) +
            sapply(rho, function(x)
                   2 * c(determinant(update(C1, nWC, 1/x))$modulus)))
all.equal(MJ, MJ1)
## Don't show: 
stopifnot( all.equal(MJ, MJ1) )
## End(Don't show)

C2 <- Cholesky(nWC, super = TRUE, Imult = 2)
system.time(MJ2 <- n * log(rho) +
            sapply(rho, function(x)
                   2 * c(determinant(update(C2, nWC, 1/x))$modulus)))
all.equal(MJ, MJ2)  ## Don't show: 
stopifnot(all.equal(MJ, MJ2))
## End(Don't show)
system.time(MJ3 <- n * log(rho) + Matrix:::ldetL2up(C1, nWC, 1/rho))
stopifnot(all.equal(MJ, MJ3))
system.time(MJ4 <- n * log(rho) + Matrix:::ldetL2up(C2, nWC, 1/rho))
stopifnot(all.equal(MJ, MJ4))



cleanEx()
nameEx("Xtrct-methods")
### * Xtrct-methods

flush(stderr()); flush(stdout())

### Name: [-methods
### Title: Methods for "[": Extraction or Subsetting in Package 'Matrix'
### Aliases: [-methods [,CsparseMatrix,index,index,logical-method
###   [,CsparseMatrix,index,missing,logical-method
###   [,CsparseMatrix,missing,index,logical-method
###   [,Matrix,ANY,ANY,ANY-method [,Matrix,index,index,missing-method
###   [,Matrix,index,missing,missing-method
###   [,Matrix,lMatrix,missing,missing-method
###   [,Matrix,logical,missing,missing-method
###   [,Matrix,matrix,missing,ANY-method
###   [,Matrix,matrix,missing,missing-method
###   [,Matrix,missing,index,missing-method
###   [,Matrix,missing,missing,ANY-method
###   [,Matrix,missing,missing,logical-method
###   [,Matrix,missing,missing,missing-method
###   [,Matrix,nMatrix,missing,missing-method
###   [,TsparseMatrix,index,index,logical-method
###   [,TsparseMatrix,index,missing,logical-method
###   [,TsparseMatrix,missing,index,logical-method
###   [,abIndex,index,ANY,ANY-method
###   [,denseMatrix,index,index,logical-method
###   [,denseMatrix,index,missing,logical-method
###   [,denseMatrix,matrix,missing,ANY-method
###   [,denseMatrix,matrix,missing,missing-method
###   [,denseMatrix,missing,index,logical-method
###   [,diagonalMatrix,index,index,logical-method
###   [,diagonalMatrix,index,missing,logical-method
###   [,diagonalMatrix,missing,index,logical-method
###   [,indMatrix,index,missing,logical-method
###   [,packedMatrix,NULL,NULL,logical-method
###   [,packedMatrix,NULL,NULL,missing-method
###   [,packedMatrix,NULL,index,logical-method
###   [,packedMatrix,NULL,index,missing-method
###   [,packedMatrix,NULL,missing,logical-method
###   [,packedMatrix,NULL,missing,missing-method
###   [,packedMatrix,index,NULL,logical-method
###   [,packedMatrix,index,NULL,missing-method
###   [,packedMatrix,index,index,logical-method
###   [,packedMatrix,index,index,missing-method
###   [,packedMatrix,index,missing,logical-method
###   [,packedMatrix,index,missing,missing-method
###   [,packedMatrix,lMatrix,NULL,logical-method
###   [,packedMatrix,lMatrix,NULL,missing-method
###   [,packedMatrix,lMatrix,index,logical-method
###   [,packedMatrix,lMatrix,index,missing-method
###   [,packedMatrix,lMatrix,missing,logical-method
###   [,packedMatrix,lMatrix,missing,missing-method
###   [,packedMatrix,matrix,NULL,logical-method
###   [,packedMatrix,matrix,NULL,missing-method
###   [,packedMatrix,matrix,index,logical-method
###   [,packedMatrix,matrix,index,missing-method
###   [,packedMatrix,matrix,missing,logical-method
###   [,packedMatrix,matrix,missing,missing-method
###   [,packedMatrix,missing,NULL,logical-method
###   [,packedMatrix,missing,NULL,missing-method
###   [,packedMatrix,missing,index,logical-method
###   [,packedMatrix,missing,index,missing-method
###   [,packedMatrix,missing,missing,logical-method
###   [,packedMatrix,missing,missing,missing-method
###   [,sparseMatrix,index,index,logical-method
###   [,sparseMatrix,index,missing,logical-method
###   [,sparseMatrix,missing,index,logical-method
###   [,sparseVector,index,ANY,ANY-method
###   [,sparseVector,lsparseVector,ANY,ANY-method
###   [,sparseVector,nsparseVector,ANY,ANY-method
### Keywords: methods array

### ** Examples

str(m <- Matrix(round(rnorm(7*4),2), nrow = 7))
stopifnot(identical(m, m[]))
m[2, 3]   # simple number
m[2, 3:4] # simple numeric of length 2
m[2, 3:4, drop=FALSE] # sub matrix of class 'dgeMatrix'
## rows or columns only:
m[1,]     # first row, as simple numeric vector
m[,1:2]   # sub matrix of first two columns

showMethods("[", inherited = FALSE)



cleanEx()
nameEx("abIndex-class")
### * abIndex-class

flush(stderr()); flush(stdout())

### Name: abIndex-class
### Title: Class "abIndex" of Abstract Index Vectors
### Aliases: abIndex-class seqMat-class Arith,abIndex,abIndex-method
###   Arith,abIndex,numLike-method Arith,numLike,abIndex-method
###   Ops,ANY,abIndex-method Ops,abIndex,ANY-method
###   Ops,abIndex,abIndex-method Summary,abIndex-method
###   as.integer,abIndex-method as.numeric,abIndex-method
###   as.vector,abIndex-method coerce,abIndex,integer-method
###   coerce,abIndex,numeric-method coerce,abIndex,seqMat-method
###   coerce,abIndex,vector-method coerce,logical,abIndex-method
###   coerce,numeric,abIndex-method drop,abIndex-method
###   length,abIndex-method show,abIndex-method
###   coerce,numeric,seqMat-method coerce,seqMat,abIndex-method
###   coerce,seqMat,numeric-method
### Keywords: classes

### ** Examples

showClass("abIndex")
ii <- c(-3:40, 20:70)
str(ai <- as(ii, "abIndex"))# note
ai # -> show() method

stopifnot(identical(-3:20,
                    as(abIseq1(-3,20), "vector")))



cleanEx()
nameEx("abIseq")
### * abIseq

flush(stderr()); flush(stdout())

### Name: abIseq
### Title: Sequence Generation of "abIndex", Abstract Index Vectors
### Aliases: abIseq abIseq1 c.abIndex
### Keywords: manip classes

### ** Examples

stopifnot(identical(-3:20,
                    as(abIseq1(-3,20), "vector")))

try( ## (arithmetic) not yet implemented
abIseq(1, 50, by = 3)
)




cleanEx()
nameEx("all-methods")
### * all-methods

flush(stderr()); flush(stdout())

### Name: all-methods
### Title: "Matrix" Methods for Functions all() and any()
### Aliases: all-methods any-methods all,Matrix-method all,ddiMatrix-method
###   all,ldiMatrix-method all,lsparseMatrix-method
###   all,nsparseMatrix-method any,Matrix-method any,ddiMatrix-method
###   any,lMatrix-method any,ldiMatrix-method any,nsparseMatrix-method
### Keywords: methods

### ** Examples

M <- Matrix(1:12 +0, 3,4)
all(M >= 1) # TRUE
any(M < 0 ) # FALSE
MN <- M; MN[2,3] <- NA; MN
all(MN >= 0) # NA
any(MN <  0) # NA
any(MN <  0, na.rm = TRUE) # -> FALSE
## Don't show: 
sM <- as(MN, "sparseMatrix")
stopifnot(all(M >= 1), !any(M < 0),
          all.equal((sM >= 1), as(MN >= 1, "sparseMatrix")),
          ## MN:
          any(MN < 2), !all(MN < 5),
          is.na(all(MN >= 0)), is.na(any(MN < 0)),
          all(MN >= 0, na.rm=TRUE), !any(MN < 0, na.rm=TRUE),
          ## same for sM :
          any(sM < 2), !all(sM < 5),
          is.na(all(sM >= 0)), is.na(any(sM < 0)),
          all(sM >= 0, na.rm=TRUE), !any(sM < 0, na.rm=TRUE)
         )
## End(Don't show)



cleanEx()
nameEx("all.equal-methods")
### * all.equal-methods

flush(stderr()); flush(stdout())

### Name: all.equal-methods
### Title: Matrix Package Methods for Function all.equal()
### Aliases: all.equal-methods all.equal,ANY,Matrix-method
###   all.equal,ANY,sparseMatrix-method all.equal,ANY,sparseVector-method
###   all.equal,Matrix,ANY-method all.equal,Matrix,Matrix-method
###   all.equal,abIndex,abIndex-method all.equal,abIndex,numLike-method
###   all.equal,numLike,abIndex-method all.equal,sparseMatrix,ANY-method
###   all.equal,sparseMatrix,sparseMatrix-method
###   all.equal,sparseMatrix,sparseVector-method
###   all.equal,sparseVector,ANY-method
###   all.equal,sparseVector,sparseMatrix-method
###   all.equal,sparseVector,sparseVector-method
### Keywords: methods arith

### ** Examples

showMethods("all.equal")

(A <- spMatrix(3,3, i= c(1:3,2:1), j=c(3:1,1:2), x = 1:5))
ex <- expand(lu. <- lu(A))
stopifnot( all.equal(as(A[lu.@p + 1L, lu.@q + 1L], "CsparseMatrix"),
                     lu.@L %*% lu.@U),
           with(ex, all.equal(as(P %*% A %*% Q, "CsparseMatrix"),
                              L %*% U)),
           with(ex, all.equal(as(A, "CsparseMatrix"),
                              t(P) %*% L %*% U %*% t(Q))))



cleanEx()
nameEx("atomicVector-class")
### * atomicVector-class

flush(stderr()); flush(stdout())

### Name: atomicVector-class
### Title: Virtual Class "atomicVector" of Atomic Vectors
### Aliases: atomicVector-class Ops,atomicVector,sparseVector-method
###   cbind2,atomicVector,Matrix-method
###   cbind2,atomicVector,ddiMatrix-method
###   cbind2,atomicVector,ldiMatrix-method
###   coerce,atomicVector,dsparseVector-method
###   coerce,atomicVector,sparseVector-method
###   rbind2,atomicVector,Matrix-method
###   rbind2,atomicVector,ddiMatrix-method
###   rbind2,atomicVector,ldiMatrix-method
### Keywords: classes

### ** Examples

showClass("atomicVector")



cleanEx()
nameEx("band")
### * band

flush(stderr()); flush(stdout())

### Name: band
### Title: Extract bands of a matrix
### Aliases: band band-methods triu triu-methods tril tril-methods
###   band,CsparseMatrix-method band,RsparseMatrix-method
###   band,TsparseMatrix-method band,denseMatrix-method
###   band,diagonalMatrix-method band,indMatrix-method band,matrix-method
###   triu,CsparseMatrix-method triu,RsparseMatrix-method
###   triu,TsparseMatrix-method triu,denseMatrix-method
###   triu,diagonalMatrix-method triu,indMatrix-method triu,matrix-method
###   tril,CsparseMatrix-method tril,RsparseMatrix-method
###   tril,TsparseMatrix-method tril,denseMatrix-method
###   tril,diagonalMatrix-method tril,indMatrix-method tril,matrix-method
### Keywords: methods algebra

### ** Examples

## A random sparse matrix :
set.seed(7)
m <- matrix(0, 5, 5)
m[sample(length(m), size = 14)] <- rep(1:9, length=14)
(mm <- as(m, "CsparseMatrix"))

tril(mm)        # lower triangle
tril(mm, -1)    # strict lower triangle
triu(mm,  1)    # strict upper triangle
band(mm, -1, 2) # general band
(m5 <- Matrix(rnorm(25), ncol = 5))
tril(m5)        # lower triangle
tril(m5, -1)    # strict lower triangle
triu(m5, 1)     # strict upper triangle
band(m5, -1, 2) # general band
(m65 <- Matrix(rnorm(30), ncol = 5))  # not square
triu(m65)       # result not "dtrMatrix" unless square
(sm5 <- crossprod(m65)) # symmetric
   band(sm5, -1, 1)# "dsyMatrix": symmetric band preserves symmetry property
as(band(sm5, -1, 1), "sparseMatrix")# often preferable
(sm <- round(crossprod(triu(mm/2)))) # sparse symmetric ("dsC*")
band(sm, -1,1) # remains "dsC", *however*
band(sm, -2,1) # -> "dgC"
## Don't show: 
 ## this uses special methods
(x.x <- crossprod(mm))
tril(x.x)
xx <- tril(x.x) + triu(x.x, 1) ## the same as x.x (but stored differently):
txx <- t(as(xx, "symmetricMatrix"))
stopifnot(identical(triu(x.x), t(tril(x.x))),
	  identical(class(x.x), class(txx)),
	  identical(as(x.x, "generalMatrix"), as(txx, "generalMatrix")))
## End(Don't show)



cleanEx()
nameEx("bandSparse")
### * bandSparse

flush(stderr()); flush(stdout())

### Name: bandSparse
### Title: Construct Sparse Banded Matrix from (Sup-/Super-) Diagonals
### Aliases: bandSparse
### Keywords: array algebra

### ** Examples

diags <- list(1:30, 10*(1:20), 100*(1:20))
s1 <- bandSparse(13, k = -c(0:2, 6), diag = c(diags, diags[2]), symm=TRUE)
s1
s2 <- bandSparse(13, k =  c(0:2, 6), diag = c(diags, diags[2]), symm=TRUE)
stopifnot(identical(s1, t(s2)), is(s1,"dsCMatrix"))

## a pattern Matrix of *full* (sub-)diagonals:
bk <- c(0:4, 7,9)
(s3 <- bandSparse(30, k = bk, symm = TRUE))

## If you want a pattern matrix, but with "sparse"-diagonals,
## you currently need to go via logical sparse:
lLis <- lapply(list(rpois(20, 2), rpois(20,1), rpois(20,3))[c(1:3,2:3,3:2)],
               as.logical)
(s4 <- bandSparse(20, k = bk, symm = TRUE, diag = lLis))
(s4. <- as(drop0(s4), "nsparseMatrix"))

n <- 1e4
bk <- c(0:5, 7,11)
bMat <- matrix(1:8, n, 8, byrow=TRUE)
bLis <- as.data.frame(bMat)
B  <- bandSparse(n, k = bk, diag = bLis)
Bs <- bandSparse(n, k = bk, diag = bLis, symmetric=TRUE)
B [1:15, 1:30]
Bs[1:15, 1:30]
## can use a list *or* a matrix for specifying the diagonals:
stopifnot(identical(B,  bandSparse(n, k = bk, diag = bMat)),
	  identical(Bs, bandSparse(n, k = bk, diag = bMat, symmetric=TRUE))
          , inherits(B, "dtCMatrix") # triangular!
)



cleanEx()
nameEx("bdiag")
### * bdiag

flush(stderr()); flush(stdout())

### Name: bdiag
### Title: Construct a Block Diagonal Matrix
### Aliases: bdiag .bdiag
### Keywords: array

### ** Examples

bdiag(matrix(1:4, 2), diag(3))
## combine "Matrix" class and traditional matrices:
bdiag(Diagonal(2), matrix(1:3, 3,4), diag(3:2))

mlist <- list(1, 2:3, diag(x=5:3), 27, cbind(1,3:6), 100:101)
bdiag(mlist)
stopifnot(identical(bdiag(mlist), 
                    bdiag(lapply(mlist, as.matrix))))

ml <- c(as(matrix((1:24)%% 11 == 0, 6,4),"nMatrix"),
        rep(list(Diagonal(2, x=TRUE)), 3))
mln <- c(ml, Diagonal(x = 1:3))
stopifnot(is(bdiag(ml), "lsparseMatrix"),
          is(bdiag(mln),"dsparseMatrix") )

## random (diagonal-)block-triangular matrices:
rblockTri <- function(nb, max.ni, lambda = 3) {
   .bdiag(replicate(nb, {
         n <- sample.int(max.ni, 1)
         tril(Matrix(rpois(n*n, lambda=lambda), n,n)) }))
}

(T4 <- rblockTri(4, 10, lambda = 1))
image(T1 <- rblockTri(12, 20))


##' Fast version of Matrix :: .bdiag() -- for the case of *many*  (k x k) matrices:
##' @param lmat list(<mat1>, <mat2>, ....., <mat_N>)  where each mat_j is a  k x k 'matrix'
##' @return a sparse (N*k x N*k) matrix of class  \code{"\linkS4class{dgCMatrix}"}.
bdiag_m <- function(lmat) {
    ## Copyright (C) 2016 Martin Maechler, ETH Zurich
    if(!length(lmat)) return(new("dgCMatrix"))
    stopifnot(is.list(lmat), is.matrix(lmat[[1]]),
              (k <- (d <- dim(lmat[[1]]))[1]) == d[2], # k x k
              all(vapply(lmat, dim, integer(2)) == k)) # all of them
    N <- length(lmat)
    if(N * k > .Machine$integer.max)
        stop("resulting matrix too large; would be  M x M, with M=", N*k)
    M <- as.integer(N * k)
    ## result: an   M x M  matrix
    new("dgCMatrix", Dim = c(M,M),
        ## 'i :' maybe there's a faster way (w/o matrix indexing), but elegant?
        i = as.vector(matrix(0L:(M-1L), nrow=k)[, rep(seq_len(N), each=k)]),
        p = k * 0L:M,
        x = as.double(unlist(lmat, recursive=FALSE, use.names=FALSE)))
}

l12 <- replicate(12, matrix(rpois(16, lambda = 6.4), 4,4), simplify=FALSE)
dim(T12 <- bdiag_m(l12))# 48 x 48
T12[1:20, 1:20]



cleanEx()
nameEx("boolean-matprod")
### * boolean-matprod

flush(stderr()); flush(stdout())

### Name: %&%-methods
### Title: Boolean Arithmetic Matrix Products: '%&%' and Methods
### Aliases: %&% %&%-methods %&%,ANY,ANY-method %&%,ANY,Matrix-method
###   %&%,ANY,matrix-method %&%,CsparseMatrix,RsparseMatrix-method
###   %&%,CsparseMatrix,TsparseMatrix-method
###   %&%,CsparseMatrix,diagonalMatrix-method
###   %&%,CsparseMatrix,mMatrix-method %&%,Matrix,ANY-method
###   %&%,Matrix,Matrix-method %&%,Matrix,indMatrix-method
###   %&%,Matrix,pMatrix-method %&%,RsparseMatrix,CsparseMatrix-method
###   %&%,RsparseMatrix,RsparseMatrix-method
###   %&%,RsparseMatrix,TsparseMatrix-method
###   %&%,RsparseMatrix,diagonalMatrix-method
###   %&%,RsparseMatrix,mMatrix-method
###   %&%,TsparseMatrix,CsparseMatrix-method
###   %&%,TsparseMatrix,RsparseMatrix-method
###   %&%,TsparseMatrix,TsparseMatrix-method
###   %&%,TsparseMatrix,diagonalMatrix-method
###   %&%,TsparseMatrix,mMatrix-method %&%,denseMatrix,denseMatrix-method
###   %&%,denseMatrix,diagonalMatrix-method
###   %&%,diagonalMatrix,CsparseMatrix-method
###   %&%,diagonalMatrix,RsparseMatrix-method
###   %&%,diagonalMatrix,TsparseMatrix-method
###   %&%,diagonalMatrix,denseMatrix-method
###   %&%,diagonalMatrix,diagonalMatrix-method
###   %&%,diagonalMatrix,matrix-method %&%,indMatrix,Matrix-method
###   %&%,indMatrix,indMatrix-method %&%,indMatrix,matrix-method
###   %&%,indMatrix,pMatrix-method %&%,mMatrix,CsparseMatrix-method
###   %&%,mMatrix,RsparseMatrix-method %&%,mMatrix,TsparseMatrix-method
###   %&%,mMatrix,sparseMatrix-method %&%,mMatrix,sparseVector-method
###   %&%,matrix,ANY-method %&%,matrix,diagonalMatrix-method
###   %&%,matrix,indMatrix-method %&%,matrix,matrix-method
###   %&%,matrix,pMatrix-method %&%,nCsparseMatrix,nCsparseMatrix-method
###   %&%,nCsparseMatrix,nsparseMatrix-method %&%,nMatrix,nMatrix-method
###   %&%,nMatrix,nsparseMatrix-method
###   %&%,nsparseMatrix,nCsparseMatrix-method
###   %&%,nsparseMatrix,nMatrix-method
###   %&%,nsparseMatrix,nsparseMatrix-method
###   %&%,numLike,sparseVector-method %&%,sparseMatrix,mMatrix-method
###   %&%,sparseMatrix,sparseMatrix-method %&%,sparseVector,mMatrix-method
###   %&%,sparseVector,numLike-method %&%,sparseVector,sparseVector-method
### Keywords: methods

### ** Examples

set.seed(7)
L <- Matrix(rnorm(20) > 1,    4,5)
(N <- as(L, "nMatrix"))
L. <- L; L.[1:2,1] <- TRUE; L.@x[1:2] <- FALSE; L. # has "zeros" to drop0()
D <- Matrix(round(rnorm(30)), 5,6) # -> values in -1:1 (for this seed)
L %&% D
stopifnot(identical(L %&% D, N %&% D),
          all(L %&% D == as((L %*% abs(D)) > 0, "sparseMatrix")))

## cross products , possibly with  boolArith = TRUE :
crossprod(N)     # -> sparse patter'n' (TRUE/FALSE : boolean arithmetic)
crossprod(N  +0) # -> numeric Matrix (with same "pattern")
stopifnot(all(crossprod(N) == t(N) %&% N),
          identical(crossprod(N), crossprod(N +0, boolArith=TRUE)),
          identical(crossprod(L), crossprod(N   , boolArith=FALSE)))
crossprod(D, boolArith =  TRUE) # pattern: "nsCMatrix"
crossprod(L, boolArith =  TRUE) #  ditto
crossprod(L, boolArith = FALSE) # numeric: "dsCMatrix"



cleanEx()
nameEx("cBind")
### * cBind

flush(stderr()); flush(stdout())

### Name: cBind
### Title: 'cbind()' and 'rbind()' recursively built on cbind2/rbind2
### Aliases: cbind2,denseMatrix,sparseMatrix-method
###   cbind2,sparseMatrix,denseMatrix-method
###   rbind2,denseMatrix,sparseMatrix-method
###   rbind2,sparseMatrix,denseMatrix-method
### Keywords: array manip

### ** Examples

(a <- matrix(c(2:1,1:2), 2,2))

(M1 <- cbind(0, rbind(a, 7))) # a traditional matrix

D <- Diagonal(2)
(M2 <- cbind(4, a, D, -1, D, 0)) # a sparse Matrix

stopifnot(validObject(M2), inherits(M2, "sparseMatrix"),
          dim(M2) == c(2,9))



cleanEx()
nameEx("chol")
### * chol

flush(stderr()); flush(stdout())

### Name: chol
### Title: The Cholesky Decomposition - 'Matrix' S4 Generic and Methods
### Aliases: chol chol-methods chol,diagonalMatrix-method
###   chol,dgCMatrix-method chol,dgRMatrix-method chol,dgTMatrix-method
###   chol,dgeMatrix-method chol,dsCMatrix-method chol,dsRMatrix-method
###   chol,dsTMatrix-method chol,dspMatrix-method chol,dsyMatrix-method
###   chol,generalMatrix-method chol,symmetricMatrix-method
###   chol,triangularMatrix-method
### Keywords: algebra array

### ** Examples

showMethods(chol, inherited = FALSE) # show different methods

sy2 <- new("dsyMatrix", Dim = as.integer(c(2,2)), x = c(14, NA,32,77))
(c2 <- chol(sy2))#-> "Cholesky" matrix
stopifnot(all.equal(c2, chol(as(sy2, "dpoMatrix")), tolerance= 1e-13))
str(c2)

## An example where chol() can't work
(sy3 <- new("dsyMatrix", Dim = as.integer(c(2,2)), x = c(14, -1, 2, -7)))
try(chol(sy3)) # error, since it is not positive definite

## A sparse example --- exemplifying 'pivot'
(mm <- toeplitz(as(c(10, 0, 1, 0, 3), "sparseVector"))) # 5 x 5
(R <- chol(mm)) ## default:  pivot = FALSE
R2 <- chol(mm, pivot=FALSE)
stopifnot( identical(R, R2), all.equal(crossprod(R), mm) )
(R. <- chol(mm, pivot=TRUE))# nice band structure,
## but of course crossprod(R.) is *NOT* equal to mm
## --> see Cholesky() and its examples, for the pivot structure & factorization
stopifnot(all.equal(sqrt(det(mm)), det(R)),
          all.equal(prod(diag(R)), det(R)),
          all.equal(prod(diag(R.)), det(R)))

## a second, even sparser example:
(M2 <- toeplitz(as(c(1,.5, rep(0,12), -.1), "sparseVector")))
c2 <- chol(M2)
C2 <- chol(M2, pivot=TRUE)
## For the experts, check the caching of the factorizations:
ff <- M2@factors[["spdCholesky"]]
FF <- M2@factors[["sPdCholesky"]]
L1 <- as(ff, "Matrix")# pivot=FALSE: no perm.
L2 <- as(FF, "Matrix"); P2 <- as(FF, "pMatrix")
stopifnot(identical(t(L1), c2),
          all.equal(t(L2), C2, tolerance=0),#-- why not identical()?
          all.equal(M2, tcrossprod(L1)),             # M = LL'
          all.equal(M2, crossprod(crossprod(L2, P2)))# M = P'L L'P
         )



cleanEx()
nameEx("chol2inv-methods")
### * chol2inv-methods

flush(stderr()); flush(stdout())

### Name: chol2inv-methods
### Title: Inverse from Choleski or QR Decomposition - Matrix Methods
### Aliases: chol2inv-methods chol2inv,ANY-method chol2inv,CHMfactor-method
###   chol2inv,denseMatrix-method chol2inv,diagonalMatrix-method
###   chol2inv,dtrMatrix-method chol2inv,sparseMatrix-method
### Keywords: methods algebra

### ** Examples

(M  <- Matrix(cbind(1, 1:3, c(1,3,7))))
(cM <- chol(M)) # a "Cholesky" object, inheriting from "dtrMatrix"
chol2inv(cM) %*% M # the identity
stopifnot(all(chol2inv(cM) %*% M - Diagonal(nrow(M))) < 1e-10)



cleanEx()
nameEx("colSums")
### * colSums

flush(stderr()); flush(stdout())

### Name: colSums
### Title: Form Row and Column Sums and Means
### Aliases: colSums colMeans rowSums rowMeans colSums,CsparseMatrix-method
###   colSums,RsparseMatrix-method colSums,TsparseMatrix-method
###   colSums,denseMatrix-method colSums,diagonalMatrix-method
###   colSums,indMatrix-method colMeans,CsparseMatrix-method
###   colMeans,RsparseMatrix-method colMeans,TsparseMatrix-method
###   colMeans,denseMatrix-method colMeans,diagonalMatrix-method
###   colMeans,indMatrix-method rowSums,CsparseMatrix-method
###   rowSums,RsparseMatrix-method rowSums,TsparseMatrix-method
###   rowSums,denseMatrix-method rowSums,diagonalMatrix-method
###   rowSums,indMatrix-method rowMeans,CsparseMatrix-method
###   rowMeans,RsparseMatrix-method rowMeans,TsparseMatrix-method
###   rowMeans,denseMatrix-method rowMeans,diagonalMatrix-method
###   rowMeans,indMatrix-method
### Keywords: array algebra arith

### ** Examples

(M <- bdiag(Diagonal(2), matrix(1:3, 3,4), diag(3:2))) # 7 x 8
colSums(M)
d <- Diagonal(10, c(0,0,10,0,2,rep(0,5)))
MM <- kronecker(d, M)
dim(MM) # 70 80
length(MM@x) # 160, but many are '0' ; drop those:
MM <- drop0(MM)
length(MM@x) # 32
  cm <- colSums(MM)
(scm <- colSums(MM, sparseResult = TRUE))
stopifnot(is(scm, "sparseVector"),
          identical(cm, as.numeric(scm)))
rowSums (MM, sparseResult = TRUE) # 14 of 70 are not zero
colMeans(MM, sparseResult = TRUE) # 16 of 80 are not zero
## Since we have no 'NA's, these two are equivalent :
stopifnot(identical(rowMeans(MM, sparseResult = TRUE),
                    rowMeans(MM, sparseResult = TRUE, na.rm = TRUE)),
	  rowMeans(Diagonal(16)) == 1/16,
	  colSums(Diagonal(7)) == 1)

## dimnames(x) -->  names( <value> ) :
dimnames(M) <- list(paste0("r", 1:7), paste0("V",1:8))
M
colSums(M)
rowMeans(M)
## Assertions :
stopifnot(all.equal(colSums(M),
		    setNames(c(1,1,6,6,6,6,3,2), colnames(M))),
	  all.equal(rowMeans(M), structure(c(1,1,4,8,12,3,2) / 8,
					   .Names = paste0("r", 1:7))))



cleanEx()
nameEx("condest")
### * condest

flush(stderr()); flush(stdout())

### Name: condest
### Title: Compute Approximate CONDition number and 1-Norm of (Large)
###   Matrices
### Aliases: condest onenormest

### ** Examples

data(KNex)
mtm <- with(KNex, crossprod(mm))
system.time(ce <- condest(mtm))
sum(abs(ce$v)) ## || v ||_1  == 1
## Prove that  || A v || = || A || / est  (as ||v|| = 1):
stopifnot(all.equal(norm(mtm %*% ce$v),
                    norm(mtm) / ce$est))

## reciprocal
1 / ce$est
system.time(rc <- rcond(mtm)) # takes ca  3 x  longer
rc
all.equal(rc, 1/ce$est) # TRUE -- the approxmation was good

one <- onenormest(mtm)
str(one) ## est = 12.3
## the maximal column:
which(one$v == 1) # mostly 4, rarely 1, depending on random seed



cleanEx()
nameEx("dMatrix-class")
### * dMatrix-class

flush(stderr()); flush(stdout())

### Name: dMatrix-class
### Title: (Virtual) Class "dMatrix" of "double" Matrices
### Aliases: dMatrix-class lMatrix-class Compare,dMatrix,logical-method
###   Compare,dMatrix,numeric-method Compare,logical,dMatrix-method
###   Compare,numeric,dMatrix-method Logic,dMatrix,logical-method
###   Logic,dMatrix,numeric-method Logic,dMatrix,sparseVector-method
###   Logic,logical,dMatrix-method Logic,numeric,dMatrix-method
###   Math2,dMatrix-method Ops,dMatrix,dMatrix-method
###   Ops,dMatrix,ddiMatrix-method Ops,dMatrix,lMatrix-method
###   Ops,dMatrix,ldiMatrix-method Ops,dMatrix,nMatrix-method
###   coerce,matrix,dMatrix-method coerce,numLike,dMatrix-method
###   zapsmall,dMatrix-method Arith,lMatrix,numeric-method
###   Arith,lMatrix,logical-method Arith,logical,lMatrix-method
###   Arith,numeric,lMatrix-method Compare,lMatrix,logical-method
###   Compare,lMatrix,numeric-method Compare,logical,lMatrix-method
###   Compare,numeric,lMatrix-method Logic,lMatrix,logical-method
###   Logic,lMatrix,numeric-method Logic,lMatrix,sparseVector-method
###   Logic,logical,lMatrix-method Logic,numeric,lMatrix-method
###   Ops,lMatrix,dMatrix-method Ops,lMatrix,lMatrix-method
###   Ops,lMatrix,nMatrix-method Ops,lMatrix,numeric-method
###   Ops,numeric,lMatrix-method Summary,lMatrix-method
###   coerce,matrix,lMatrix-method coerce,numLike,lMatrix-method
### Keywords: classes algebra

### ** Examples

 showClass("dMatrix")

 set.seed(101)
 round(Matrix(rnorm(28), 4,7), 2)
 M <- Matrix(rlnorm(56, sd=10), 4,14)
 (M. <- zapsmall(M))
 table(as.logical(M. == 0))



cleanEx()
nameEx("ddenseMatrix-class")
### * ddenseMatrix-class

flush(stderr()); flush(stdout())

### Name: ddenseMatrix-class
### Title: Virtual Class "ddenseMatrix" of Numeric Dense Matrices
### Aliases: ddenseMatrix-class &,ddenseMatrix,ddiMatrix-method
###   &,ddenseMatrix,ldiMatrix-method *,ddenseMatrix,ddiMatrix-method
###   *,ddenseMatrix,ldiMatrix-method Arith,ddenseMatrix,logical-method
###   Arith,ddenseMatrix,numeric-method
###   Arith,ddenseMatrix,sparseVector-method
###   Arith,logical,ddenseMatrix-method Arith,numeric,ddenseMatrix-method
###   Math,ddenseMatrix-method Summary,ddenseMatrix-method
###   ^,ddenseMatrix,ddiMatrix-method ^,ddenseMatrix,ldiMatrix-method
###   coerce,matrix,ddenseMatrix-method coerce,numLike,ddenseMatrix-method
###   log,ddenseMatrix-method
### Keywords: classes

### ** Examples

showClass("ddenseMatrix")

showMethods(class = "ddenseMatrix", where = "package:Matrix")



cleanEx()
nameEx("ddiMatrix-class")
### * ddiMatrix-class

flush(stderr()); flush(stdout())

### Name: ddiMatrix-class
### Title: Class "ddiMatrix" of Diagonal Numeric Matrices
### Aliases: ddiMatrix-class %%,ddiMatrix,Matrix-method
###   %%,ddiMatrix,ddenseMatrix-method %%,ddiMatrix,ldenseMatrix-method
###   %%,ddiMatrix,ndenseMatrix-method %/%,ddiMatrix,Matrix-method
###   %/%,ddiMatrix,ddenseMatrix-method %/%,ddiMatrix,ldenseMatrix-method
###   %/%,ddiMatrix,ndenseMatrix-method &,ddiMatrix,Matrix-method
###   &,ddiMatrix,ddenseMatrix-method &,ddiMatrix,ldenseMatrix-method
###   &,ddiMatrix,ndenseMatrix-method *,ddiMatrix,Matrix-method
###   *,ddiMatrix,ddenseMatrix-method *,ddiMatrix,ldenseMatrix-method
###   *,ddiMatrix,ndenseMatrix-method -,ddiMatrix,missing-method
###   /,ddiMatrix,Matrix-method /,ddiMatrix,ddenseMatrix-method
###   /,ddiMatrix,ldenseMatrix-method /,ddiMatrix,ndenseMatrix-method
###   Arith,ddiMatrix,logical-method Arith,ddiMatrix,numeric-method
###   Arith,logical,ddiMatrix-method Arith,numeric,ddiMatrix-method
###   Ops,ANY,ddiMatrix-method Ops,ddiMatrix,ANY-method
###   Ops,ddiMatrix,Matrix-method Ops,ddiMatrix,dMatrix-method
###   Ops,ddiMatrix,ddiMatrix-method Ops,ddiMatrix,ldiMatrix-method
###   Ops,ddiMatrix,logical-method Ops,ddiMatrix,numeric-method
###   Ops,ddiMatrix,sparseMatrix-method Summary,ddiMatrix-method
###   as.numeric,ddiMatrix-method cbind2,ddiMatrix,atomicVector-method
###   cbind2,ddiMatrix,matrix-method cbind2,matrix,ddiMatrix-method
###   prod,ddiMatrix-method rbind2,ddiMatrix,atomicVector-method
###   rbind2,ddiMatrix,matrix-method rbind2,matrix,ddiMatrix-method
###   sum,ddiMatrix-method
### Keywords: classes

### ** Examples

(d2 <- Diagonal(x = c(10,1)))
str(d2)
## slightly larger in internal size:
str(as(d2, "sparseMatrix"))

M <- Matrix(cbind(1,2:4))
M %*% d2 #> `fast' multiplication

chol(d2) # trivial
stopifnot(is(cd2 <- chol(d2), "ddiMatrix"),
          all.equal(cd2@x, c(sqrt(10),1)))



cleanEx()
nameEx("denseMatrix-class")
### * denseMatrix-class

flush(stderr()); flush(stdout())

### Name: denseMatrix-class
### Title: Virtual Class "denseMatrix" of All Dense Matrices
### Aliases: denseMatrix-class -,denseMatrix,missing-method
###   Math,denseMatrix-method as.logical,denseMatrix-method
###   as.numeric,denseMatrix-method as.vector,denseMatrix-method
###   cbind2,denseMatrix,denseMatrix-method
###   cbind2,denseMatrix,matrix-method cbind2,denseMatrix,numeric-method
###   cbind2,matrix,denseMatrix-method cbind2,numeric,denseMatrix-method
###   coerce,ANY,denseMatrix-method coerce,denseMatrix,CsparseMatrix-method
###   coerce,denseMatrix,RsparseMatrix-method
###   coerce,denseMatrix,TsparseMatrix-method
###   coerce,denseMatrix,dMatrix-method
###   coerce,denseMatrix,ddenseMatrix-method
###   coerce,denseMatrix,dsparseMatrix-method
###   coerce,denseMatrix,generalMatrix-method
###   coerce,denseMatrix,lMatrix-method
###   coerce,denseMatrix,ldenseMatrix-method
###   coerce,denseMatrix,lsparseMatrix-method
###   coerce,denseMatrix,matrix-method coerce,denseMatrix,nMatrix-method
###   coerce,denseMatrix,ndenseMatrix-method
###   coerce,denseMatrix,nsparseMatrix-method
###   coerce,denseMatrix,packedMatrix-method
###   coerce,denseMatrix,sparseMatrix-method
###   coerce,denseMatrix,unpackedMatrix-method
###   coerce,denseMatrix,vector-method coerce,matrix,denseMatrix-method
###   coerce,numLike,denseMatrix-method dim<-,denseMatrix-method
###   log,denseMatrix-method rbind2,denseMatrix,denseMatrix-method
###   rbind2,denseMatrix,matrix-method rbind2,denseMatrix,numeric-method
###   rbind2,matrix,denseMatrix-method rbind2,numeric,denseMatrix-method
###   show,denseMatrix-method
### Keywords: classes

### ** Examples

showClass("denseMatrix")



cleanEx()
nameEx("dgCMatrix-class")
### * dgCMatrix-class

flush(stderr()); flush(stdout())

### Name: dgCMatrix-class
### Title: Compressed, sparse, column-oriented numeric matrices
### Aliases: dgCMatrix-class Arith,dgCMatrix,dgCMatrix-method
###   Arith,dgCMatrix,logical-method Arith,dgCMatrix,numeric-method
###   Arith,logical,dgCMatrix-method Arith,numeric,dgCMatrix-method
###   coerce,matrix,dgCMatrix-method determinant,dgCMatrix,logical-method
### Keywords: classes algebra

### ** Examples

(m <- Matrix(c(0,0,2:0), 3,5))
str(m)
m[,1]
## Don't show: 
## regression test: this must give a validity-check error:
stopifnot(inherits(try(new("dgCMatrix", i = 0:1, p = 0:2,
                           x = c(2,3), Dim = 3:4)),
          "try-error"))
## End(Don't show)



cleanEx()
nameEx("dgTMatrix-class")
### * dgTMatrix-class

flush(stderr()); flush(stdout())

### Name: dgTMatrix-class
### Title: Sparse matrices in triplet form
### Aliases: dgTMatrix-class +,dgTMatrix,dgTMatrix-method
###   determinant,dgTMatrix,logical-method
### Keywords: classes algebra

### ** Examples

m <- Matrix(0+1:28, nrow = 4)
m[-3,c(2,4:5,7)] <- m[ 3, 1:4] <- m[1:3, 6] <- 0
(mT <- as(m, "TsparseMatrix"))
str(mT)
mT[1,]
mT[4, drop = FALSE]
stopifnot(identical(mT[lower.tri(mT)],
                    m [lower.tri(m) ]))
mT[lower.tri(mT,diag=TRUE)] <- 0
mT

## Triplet representation with repeated (i,j) entries
## *adds* the corresponding x's:
T2 <- new("dgTMatrix",
          i = as.integer(c(1,1,0,3,3)),
          j = as.integer(c(2,2,4,0,0)), x=10*1:5, Dim=4:5)
str(T2) # contains (i,j,x) slots exactly as above, but
T2 ## has only three non-zero entries, as for repeated (i,j)'s,
   ## the corresponding x's are "implicitly" added
stopifnot(nnzero(T2) == 3)



cleanEx()
nameEx("diagU2N")
### * diagU2N

flush(stderr()); flush(stdout())

### Name: diagU2N
### Title: Transform Triangular Matrices from Unit Triangular to General
###   Triangular and Back
### Aliases: diagU2N diagN2U .diagU2N .diagN2U
### Keywords: utilities classes

### ** Examples

(T <- Diagonal(7) + triu(Matrix(rpois(49, 1/4), 7,7), k = 1))
(uT <- diagN2U(T)) # "unitriangular"
(t.u <- diagN2U(10*T))# changes the diagonal!
stopifnot(all(T == uT), diag(t.u) == 1,
          identical(T, diagU2N(uT)))
T[upper.tri(T)] <- 5 # still "dtC"
T <- diagN2U(as(T,"triangularMatrix"))
dT <- as(T, "denseMatrix") # (unitriangular)
dT.n <- diagU2N(dT, checkDense = TRUE)
sT.n <- diagU2N(dT)
stopifnot(is(dT.n, "denseMatrix"), is(sT.n, "sparseMatrix"),
          dT@diag == "U", dT.n@diag == "N", sT.n@diag == "N",
          all(dT == dT.n), all(dT == sT.n))



cleanEx()
nameEx("diagonalMatrix-class")
### * diagonalMatrix-class

flush(stderr()); flush(stdout())

### Name: diagonalMatrix-class
### Title: Class "diagonalMatrix" of Diagonal Matrices
### Aliases: diagonalMatrix-class Math,diagonalMatrix-method
###   Ops,diagonalMatrix,triangularMatrix-method
###   as.logical,diagonalMatrix-method as.numeric,diagonalMatrix-method
###   as.vector,diagonalMatrix-method
###   cbind2,diagonalMatrix,sparseMatrix-method
###   coerce,diagonalMatrix,CsparseMatrix-method
###   coerce,diagonalMatrix,RsparseMatrix-method
###   coerce,diagonalMatrix,TsparseMatrix-method
###   coerce,diagonalMatrix,dMatrix-method
###   coerce,diagonalMatrix,ddenseMatrix-method
###   coerce,diagonalMatrix,denseMatrix-method
###   coerce,diagonalMatrix,dsparseMatrix-method
###   coerce,diagonalMatrix,generalMatrix-method
###   coerce,diagonalMatrix,lMatrix-method
###   coerce,diagonalMatrix,ldenseMatrix-method
###   coerce,diagonalMatrix,lsparseMatrix-method
###   coerce,diagonalMatrix,matrix-method
###   coerce,diagonalMatrix,nMatrix-method
###   coerce,diagonalMatrix,ndenseMatrix-method
###   coerce,diagonalMatrix,nsparseMatrix-method
###   coerce,diagonalMatrix,packedMatrix-method
###   coerce,diagonalMatrix,sparseVector-method
###   coerce,diagonalMatrix,symmetricMatrix-method
###   coerce,diagonalMatrix,triangularMatrix-method
###   coerce,diagonalMatrix,unpackedMatrix-method
###   coerce,diagonalMatrix,vector-method
###   coerce,matrix,diagonalMatrix-method
###   determinant,diagonalMatrix,logical-method diag,diagonalMatrix-method
###   diag<-,diagonalMatrix-method log,diagonalMatrix-method
###   print,diagonalMatrix-method rbind2,diagonalMatrix,sparseMatrix-method
###   show,diagonalMatrix-method summary,diagonalMatrix-method
###   t,diagonalMatrix-method
### Keywords: classes

### ** Examples

I5 <- Diagonal(5)
D5 <- Diagonal(x = 10*(1:5))
## trivial (but explicitly defined) methods:
stopifnot(identical(crossprod(I5), I5),
          identical(tcrossprod(I5), I5),
          identical(crossprod(I5, D5), D5),
          identical(tcrossprod(D5, I5), D5),
          identical(solve(D5), solve(D5, I5)),
          all.equal(D5, solve(solve(D5)), tolerance = 1e-12)
          )
solve(D5)# efficient as is diagonal

# an unusual way to construct a band matrix:
rbind2(cbind2(I5, D5),
       cbind2(D5, I5))



cleanEx()
nameEx("dimScale")
### * dimScale

flush(stderr()); flush(stdout())

### Name: dimScale
### Title: Scale the Rows and Columns of a Matrix
### Aliases: dimScale rowScale colScale

### ** Examples

n <- 6L
(x <- forceSymmetric(matrix(1, n, n)))
dimnames(x) <- rep.int(list(letters[seq_len(n)]), 2L)

d <- seq_len(n)
(D <- Diagonal(x = d))

(scx <- dimScale(x, d)) # symmetry and 'dimnames' kept
(mmx <- D %*% x %*% D) # symmetry and 'dimnames' lost
stopifnot(identical(unname(as(scx, "generalMatrix")), mmx))

rowScale(x, d)
colScale(x, d)



cleanEx()
nameEx("dmperm")
### * dmperm

flush(stderr()); flush(stdout())

### Name: dmperm
### Title: Dulmage-Mendelsohn Permutation / Decomposition
### Aliases: dmperm
### Keywords: algebra

### ** Examples

set.seed(17)
(S9 <- rsparsematrix(9, 9, nnz = 10, symmetric=TRUE)) # dsCMatrix
str( dm9 <- dmperm(S9) )
(S9p <- with(dm9, S9[p, q]))
## looks good, but *not* quite upper triangular; these, too:
str( dm9.0 <- dmperm(S9, seed=-1)) # non-random too.
str( dm9_1 <- dmperm(S9, seed= 1)) # a random one
## The last two permutations differ, but have the same effect!
(S9p0 <- with(dm9.0, S9[p, q])) # .. hmm ..
stopifnot(all.equal(S9p0, S9p))# same as as default, but different from the random one


set.seed(11)
(M <- triu(rsparsematrix(9,11, 1/4)))
dM <- dmperm(M); with(dM, M[p, q])
(Mp <- M[sample.int(nrow(M)), sample.int(ncol(M))])
dMp <- dmperm(Mp); with(dMp, Mp[p, q])


set.seed(7)
(n7 <- rsparsematrix(5, 12, nnz = 10, rand.x = NULL))
str( dm.7 <- dmperm(n7) )
stopifnot(exprs = {
  lengths(dm.7[1:2]) == dim(n7)
  identical(dm.7,      dmperm(as(n7, "dMatrix")))
  identical(dm.7[1:4], dmperm(n7, nAns=4))
  identical(dm.7[1:2], dmperm(n7, nAns=2))
})



cleanEx()
nameEx("dpoMatrix-class")
### * dpoMatrix-class

flush(stderr()); flush(stdout())

### Name: dpoMatrix-class
### Title: Positive Semi-definite Dense (Packed | Non-packed) Numeric
###   Matrices
### Aliases: dpoMatrix-class dppMatrix-class corMatrix-class
###   Arith,dpoMatrix,logical-method Arith,dpoMatrix,numeric-method
###   Arith,logical,dpoMatrix-method Arith,numeric,dpoMatrix-method
###   Ops,dpoMatrix,logical-method Ops,dpoMatrix,numeric-method
###   Ops,logical,dpoMatrix-method Ops,numeric,dpoMatrix-method
###   coerce,dpoMatrix,corMatrix-method coerce,dpoMatrix,dppMatrix-method
###   coerce,matrix,dpoMatrix-method determinant,dpoMatrix,logical-method
###   Arith,dppMatrix,logical-method Arith,dppMatrix,numeric-method
###   Arith,logical,dppMatrix-method Arith,numeric,dppMatrix-method
###   Ops,dppMatrix,logical-method Ops,dppMatrix,numeric-method
###   Ops,logical,dppMatrix-method Ops,numeric,dppMatrix-method
###   coerce,dppMatrix,corMatrix-method coerce,dppMatrix,dpoMatrix-method
###   coerce,matrix,dppMatrix-method determinant,dppMatrix,logical-method
###   coerce,matrix,corMatrix-method
### Keywords: classes algebra

### ** Examples

h6 <- Hilbert(6)
rcond(h6)
str(h6)
h6 * 27720 # is ``integer''
solve(h6)
str(hp6 <- as(h6, "dppMatrix"))

### Note that  as(*, "corMatrix")  *scales* the matrix
(ch6 <- as(h6, "corMatrix"))
stopifnot(all.equal(h6 * 27720, round(27720 * h6), tolerance = 1e-14),
          all.equal(ch6@sd^(-2), 2*(1:6)-1, tolerance= 1e-12))
chch <- chol(ch6)
stopifnot(identical(chch, ch6@factors$Cholesky),
          all(abs(crossprod(chch) - ch6) < 1e-10))



cleanEx()
nameEx("drop0")
### * drop0

flush(stderr()); flush(stdout())

### Name: drop0
### Title: Drop "Explicit Zeroes" from a Sparse Matrix
### Aliases: drop0
### Keywords: utilities array

### ** Examples

m <- spMatrix(10,20, i= 1:8, j=2:9, x = c(0:2,3:-1))
m
drop0(m)

## A larger example:
t5 <- new("dtCMatrix", Dim = c(5L, 5L), uplo = "L",
          x = c(10, 1, 3, 10, 1, 10, 1, 10, 10),
	  i = c(0L,2L,4L, 1L, 3L,2L,4L, 3L, 4L),
	  p = c(0L, 3L, 5L, 7:9))
TT <- kronecker(t5, kronecker(kronecker(t5,t5), t5))
IT <- solve(TT)
I. <- TT %*% IT ;  nnzero(I.) # 697 ( = 625 + 72 )
I.0 <- drop0(zapsmall(I.))
## which actually can be more efficiently achieved by
I.. <- drop0(I., tol = 1e-15)
stopifnot(all(I.0 == Diagonal(625)),
          nnzero(I..) == 625)



cleanEx()
nameEx("dsCMatrix-class")
### * dsCMatrix-class

flush(stderr()); flush(stdout())

### Name: dsCMatrix-class
### Title: Numeric Symmetric Sparse (column compressed) Matrices
### Aliases: dsCMatrix-class dsTMatrix-class
###   Arith,dsCMatrix,dsCMatrix-method
###   coerce,dsCMatrix,RsparseMatrix-method
###   determinant,dsCMatrix,logical-method
###   determinant,dsTMatrix,logical-method
### Keywords: classes algebra

### ** Examples

mm <- Matrix(toeplitz(c(10, 0, 1, 0, 3)), sparse = TRUE)
mm # automatically dsCMatrix
str(mm)
mT <- as(as(mm, "generalMatrix"), "TsparseMatrix")

## Either
(symM <- as(mT, "symmetricMatrix")) # dsT
(symC <- as(symM, "CsparseMatrix")) # dsC
## or
sT <- Matrix(mT, sparse=TRUE, forceCheck=TRUE) # dsT

sym2 <- as(symC, "TsparseMatrix")
## --> the same as 'symM', a "dsTMatrix"
## Don't show: 
stopifnot(identical(sT, symM), identical(sym2, symM),
          class(sym2) == "dsTMatrix",
	  identical(sym2[1,], sT[1,]),
	  identical(sym2[,2], sT[,2]))
## End(Don't show)



cleanEx()
nameEx("dsRMatrix-class")
### * dsRMatrix-class

flush(stderr()); flush(stdout())

### Name: dsRMatrix-class
### Title: Symmetric Sparse Compressed Row Matrices
### Aliases: dsRMatrix-class coerce,dsRMatrix,CsparseMatrix-method
###   determinant,dsRMatrix,logical-method
### Keywords: classes algebra

### ** Examples

(m0 <- new("dsRMatrix"))
m2 <- new("dsRMatrix", Dim = c(2L,2L),
          x = c(3,1), j = c(1L,1L), p = 0:2)
m2
stopifnot(colSums(as(m2, "TsparseMatrix")) == 3:4)
str(m2)
(ds2 <- forceSymmetric(diag(2))) # dsy*
dR <- as(ds2, "RsparseMatrix")
dR # dsRMatrix



cleanEx()
nameEx("dsparseMatrix-class")
### * dsparseMatrix-class

flush(stderr()); flush(stdout())

### Name: dsparseMatrix-class
### Title: Virtual Class "dsparseMatrix" of Numeric Sparse Matrices
### Aliases: dsparseMatrix-class Arith,dsparseMatrix,logical-method
###   Arith,dsparseMatrix,numeric-method Arith,logical,dsparseMatrix-method
###   Arith,numeric,dsparseMatrix-method
###   Ops,dsparseMatrix,nsparseMatrix-method Summary,dsparseMatrix-method
###   as.logical,dsparseMatrix-method as.numeric,dsparseMatrix-method
###   coerce,dsparseMatrix,lMatrix-method
###   coerce,dsparseMatrix,lsparseMatrix-method
###   coerce,dsparseMatrix,nMatrix-method
###   coerce,dsparseMatrix,nsparseMatrix-method
###   coerce,matrix,dsparseMatrix-method
###   coerce,numLike,dsparseMatrix-method
### Keywords: classes

### ** Examples

showClass("dsparseMatrix")



cleanEx()
nameEx("dsyMatrix-class")
### * dsyMatrix-class

flush(stderr()); flush(stdout())

### Name: dsyMatrix-class
### Title: Symmetric Dense (Packed or Unpacked) Numeric Matrices
### Aliases: dsyMatrix-class dspMatrix-class
###   coerce,dsyMatrix,corMatrix-method coerce,dsyMatrix,dpoMatrix-method
###   coerce,dsyMatrix,dppMatrix-method
###   determinant,dsyMatrix,logical-method
###   coerce,dspMatrix,dpoMatrix-method coerce,dspMatrix,dppMatrix-method
###   determinant,dspMatrix,logical-method
### Keywords: classes

### ** Examples

## Only upper triangular part matters (when uplo == "U" as per default)
(sy2 <- new("dsyMatrix", Dim = as.integer(c(2,2)), x = c(14, NA,32,77)))
str(t(sy2)) # uplo = "L", and the lower tri. (i.e. NA is replaced).

chol(sy2) #-> "Cholesky" matrix
(sp2 <- pack(sy2)) # a "dspMatrix"

## Coercing to dpoMatrix gives invalid object:
sy3 <- new("dsyMatrix", Dim = as.integer(c(2,2)), x = c(14, -1, 2, -7))
try(as(sy3, "dpoMatrix")) # -> error: not positive definite
## Don't show: 
tr <- try(as(sy3, "dpoMatrix"), silent=TRUE)
stopifnot(1 == grep("not a positive definite matrix",
                    as.character(tr)),
	  is(sp2, "dspMatrix"))
## End(Don't show)

## 4x4 example
m <- matrix(0,4,4); m[upper.tri(m)] <- 1:6
(sym <- m+t(m)+diag(11:14, 4))
(S1 <- pack(sym))
(S2 <- t(S1))
stopifnot(all(S1 == S2)) # equal "seen as matrix", but differ internally :
str(S1)
S2@x



cleanEx()
nameEx("dtCMatrix-class")
### * dtCMatrix-class

flush(stderr()); flush(stdout())

### Name: dtCMatrix-class
### Title: Triangular, (compressed) sparse column matrices
### Aliases: dtCMatrix-class dtTMatrix-class
###   Arith,dtCMatrix,dtCMatrix-method
### Keywords: classes algebra

### ** Examples

showClass("dtCMatrix")

showClass("dtTMatrix")
t1 <- new("dtTMatrix", x= c(3,7), i= 0:1, j=3:2, Dim= as.integer(c(4,4)))
t1
## from  0-diagonal to unit-diagonal {low-level step}:
tu <- t1 ; tu@diag <- "U"
tu
(cu <- as(tu, "CsparseMatrix"))
str(cu)# only two entries in @i and @x
stopifnot(cu@i == 1:0,
          all(2 * symmpart(cu) == Diagonal(4) + forceSymmetric(cu)))

t1[1,2:3] <- -1:-2
diag(t1) <- 10*c(1:2,3:2)
t1 # still triangular
(it1 <- solve(t1))
t1. <- solve(it1)
all(abs(t1 - t1.) < 10 * .Machine$double.eps)

## 2nd example
U5 <- new("dtCMatrix", i= c(1L, 0:3), p=c(0L,0L,0:2, 5L), Dim = c(5L, 5L),
          x = rep(1, 5), diag = "U")
U5
(iu <- solve(U5)) # contains one '0'
validObject(iu2 <- solve(U5, Diagonal(5)))# failed in earlier versions

I5 <- iu  %*% U5 # should equal the identity matrix
i5 <- iu2 %*% U5
m53 <- matrix(1:15, 5,3, dimnames=list(NULL,letters[1:3]))
asDiag <- function(M) as(drop0(M), "diagonalMatrix")
stopifnot(
   all.equal(Diagonal(5), asDiag(I5), tolerance=1e-14) ,
   all.equal(Diagonal(5), asDiag(i5), tolerance=1e-14) ,
   identical(list(NULL, dimnames(m53)[[2]]), dimnames(solve(U5, m53)))
)
## Don't show: 
i5. <- I5; colnames(i5.) <- LETTERS[11:15]
M53 <- as(m53, "denseMatrix")
stopifnot(
   identical((dns <- dimnames(solve(i5., M53))),
             dimnames(solve(as.matrix(i5.), as.matrix(M53)))) ,
   identical(dns, dimnames(solve(i5., as.matrix(M53))))
)
## End(Don't show)



cleanEx()
nameEx("dtRMatrix-class-def")
### * dtRMatrix-class-def

flush(stderr()); flush(stdout())

### Name: dtRMatrix-class
### Title: Triangular Sparse Compressed Row Matrices
### Aliases: dtRMatrix-class coerce,dtRMatrix,dgRMatrix-method
###   coerce,dtRMatrix,dsRMatrix-method coerce,dtRMatrix,dtCMatrix-method
###   coerce,dtRMatrix,dtTMatrix-method coerce,dtRMatrix,dtpMatrix-method
###   coerce,dtRMatrix,dtrMatrix-method coerce,dtRMatrix,ltRMatrix-method
###   coerce,dtRMatrix,ntRMatrix-method coerce,matrix,dtRMatrix-method
### Keywords: classes algebra

### ** Examples

(m0 <- new("dtRMatrix"))
(m2 <- new("dtRMatrix", Dim = c(2L,2L),
                        x = c(5, 1:2), p = c(0L,2:3), j= c(0:1,1L)))
str(m2)
(m3 <- as(Diagonal(2), "RsparseMatrix"))# --> dtRMatrix



cleanEx()
nameEx("dtpMatrix-class")
### * dtpMatrix-class

flush(stderr()); flush(stdout())

### Name: dtpMatrix-class
### Title: Packed Triangular Dense Matrices - "dtpMatrix"
### Aliases: dtpMatrix-class
### Keywords: classes

### ** Examples

showClass("dtrMatrix")

example("dtrMatrix-class", echo=FALSE)
(p1 <- pack(T2))
str(p1)
(pp <- pack(T))
ip1 <- solve(p1)
stopifnot(length(p1@x) == 3, length(pp@x) == 3,
          p1 @ uplo == T2 @ uplo, pp @ uplo == T @ uplo,
	  identical(t(pp), p1), identical(t(p1), pp),
	  all((l.d <- p1 - T2) == 0), is(l.d, "dtpMatrix"),
	  all((u.d <- pp - T ) == 0), is(u.d, "dtpMatrix"),
	  l.d@uplo == T2@uplo, u.d@uplo == T@uplo,
	  identical(t(ip1), solve(pp)), is(ip1, "dtpMatrix"),
	  all.equal(as(solve(p1,p1), "diagonalMatrix"), Diagonal(2)))



cleanEx()
nameEx("dtrMatrix-class")
### * dtrMatrix-class

flush(stderr()); flush(stdout())

### Name: dtrMatrix-class
### Title: Triangular, dense, numeric matrices
### Aliases: dtrMatrix-class
### Keywords: classes

### ** Examples

(m <- rbind(2:3, 0:-1))
(M <- as(m, "generalMatrix"))

(T <- as(M, "triangularMatrix")) # formally upper triangular
(T2 <- as(t(M), "triangularMatrix"))
stopifnot(T@uplo == "U", T2@uplo == "L", identical(T2, t(T)))

m <- matrix(0,4,4); m[upper.tri(m)] <- 1:6
(t1 <- Matrix(m+diag(,4)))
str(t1p <- pack(t1))
(t1pu <- diagN2U(t1p))
stopifnot(exprs = {
   inherits(t1 , "dtrMatrix"); validObject(t1)
   inherits(t1p, "dtpMatrix"); validObject(t1p)
   inherits(t1pu,"dtCMatrix"); validObject(t1pu)
   t1pu@x == 1:6
   all(t1pu == t1p)
   identical((t1pu - t1)@x, numeric())# sparse all-0
})



cleanEx()
nameEx("expand")
### * expand

flush(stderr()); flush(stdout())

### Name: expand
### Title: Expand a (Matrix) Decomposition into Factors
### Aliases: expand expand,CHMfactor-method
###   expand,MatrixFactorization-method expand,denseLU-method
###   expand,sparseLU-method
### Keywords: algebra

### ** Examples

(x <- Matrix(round(rnorm(9),2), 3, 3))
(ex <- expand(lux <- lu(x)))



cleanEx()
nameEx("expm")
### * expm

flush(stderr()); flush(stdout())

### Name: expm
### Title: Matrix Exponential
### Aliases: expm expm,Matrix-method expm,dMatrix-method
###   expm,ddiMatrix-method expm,dgeMatrix-method expm,dspMatrix-method
###   expm,dsparseMatrix-method expm,dsyMatrix-method expm,dtpMatrix-method
###   expm,dtrMatrix-method expm,matrix-method
### Keywords: algebra math

### ** Examples

(m1 <- Matrix(c(1,0,1,1), ncol = 2))
(e1 <- expm(m1)) ; e <- exp(1)
stopifnot(all.equal(e1@x, c(e,0,e,e), tolerance = 1e-15))
(m2 <- Matrix(c(-49, -64, 24, 31), ncol = 2))
(e2 <- expm(m2))
(m3 <- Matrix(cbind(0,rbind(6*diag(3),0))))# sparse!
(e3 <- expm(m3)) # upper triangular



cleanEx()
nameEx("externalFormats")
### * externalFormats

flush(stderr()); flush(stdout())

### Name: externalFormats
### Title: Read and write external matrix formats
### Aliases: readHB readMM writeMM writeMM,CsparseMatrix-method
###   writeMM,sparseMatrix-method
### Keywords: IO array algebra

### ** Examples

str(pores <- readMM(system.file("external/pores_1.mtx",
                                package = "Matrix")))
str(utm <- readHB(system.file("external/utm300.rua",
                               package = "Matrix")))
str(lundA <- readMM(system.file("external/lund_a.mtx",
                                package = "Matrix")))
str(lundA <- readHB(system.file("external/lund_a.rsa",
                                package = "Matrix")))
str(jgl009 <- ## https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/counterx/counterx.html
        readMM(system.file("external/jgl009.mtx", package = "Matrix")))
## Not run: 
##D ## NOTE: The following examples take quite some time
##D ## ----  even on a fast internet connection:
##D if(FALSE) # the URL has been corrected, but we need an un-tar step!
##D str(sm <-
##D  readHB(gzcon(url("https://www.cise.ufl.edu/research/sparse/RB/Boeing/msc00726.tar.gz"))))
## End(Not run)
data(KNex)
## Store as MatrixMarket (".mtx") file, here inside temporary dir./folder:
(MMfile <- file.path(tempdir(), "mmMM.mtx"))
writeMM(KNex$mm, file=MMfile)
file.info(MMfile)[,c("size", "ctime")] # (some confirmation of the file's)

## very simple export - in triplet format - to text file:
data(CAex)
s.CA <- summary(CAex)
s.CA # shows  (i, j, x)  [columns of a data frame]
message("writing to ", outf <- tempfile())
write.table(s.CA, file = outf, row.names=FALSE)
## and read it back -- showing off  sparseMatrix():
str(dd <- read.table(outf, header=TRUE))
## has columns (i, j, x) -> we can use via do.call() as arguments to sparseMatrix():
mm <- do.call(sparseMatrix, dd)
stopifnot(all.equal(mm, CAex, tolerance=1e-15))



cleanEx()
nameEx("facmul")
### * facmul

flush(stderr()); flush(stdout())

### Name: facmul
### Title: Multiplication by Decomposition Factors
### Aliases: facmul facmul.default
### Keywords: array algebra

### ** Examples

library(Matrix)
x <- Matrix(rnorm(9), 3, 3)
## Not run: 
##D qrx <- qr(x)                      # QR factorization of x
##D y <- rnorm(3)
##D facmul( qr(x), factor = "Q", y)   # form Q y
## End(Not run)



cleanEx()
nameEx("fastMisc")
### * fastMisc

flush(stderr()); flush(stdout())

### Name: fastMisc
### Title: "Low Level" Coercions and Methods
### Aliases: fastMisc .CR2RC .CR2T .M2diag .M2sym .M2tri .T2CR .dense2g
###   .dense2kind .dense2m .dense2sparse .dense2v .diag2dense .diag2sparse
###   .m2dense .m2sparse .sparse2dense .sparse2g .sparse2kind .sparse2m
###   .sparse2v .tCR2RC .diag.dsC .solve.dgC.chol .solve.dgC.lu
###   .solve.dgC.qr

### ** Examples

D. <- diag(x = c(1, 1, 2, 3, 5, 8))
D.0 <- Diagonal(x = c(0, 0, 0, 3, 5, 8))
S. <- toeplitz(as.double(1:6))
C. <- new("dgCMatrix", Dim = c(3L, 4L),
          p = c(0L, 1L, 1L, 1L, 3L), i = c(1L, 0L, 2L), x = c(-8, 2, 3))

stopifnot(identical(.M2tri( D.), as(D., "triangularMatrix")),
          identical(.M2sym( D.), as(D., "symmetricMatrix")),
          identical(.M2diag(D.), as(D., "diagonalMatrix")),
          identical(.sparse2kind(C., "l"),
                    as(C., "lMatrix")),
          identical(.dense2kind(.sparse2dense(C.), "l"),
                    as(as(C., "denseMatrix"), "lMatrix")),
          identical(.diag2sparse(D.0, "ntC"),
                    .dense2sparse(.diag2dense(D.0, "ntp"), "C")),
          identical(.dense2g(.diag2dense(D.0, "dsy")),
                    .sparse2dense(.sparse2g(.diag2sparse(D.0, "dsT")))),
          identical(S.,
                    .sparse2m(.m2sparse(S., ".sR"))),
          identical(S. * lower.tri(S.) + diag(1, 6L),
                    .dense2m(.m2dense(S., ".tr", "L", "U"))),
          identical(.CR2RC(C.), .T2CR(.CR2T(C.), FALSE)),
          identical(.tCR2RC(C.), .CR2RC(t(C.))))

A <- tcrossprod(C.)/6 + Diagonal(3, 1/3); A[1,2] <- 3; A
stopifnot(exprs = {
    is.numeric( x. <- c(2.2, 0, -1.2) )
    all.equal(.solve.dgC.lu(A, c(1,0,0), check=FALSE),
              Matrix(x.))
    all.equal(x., .solve.dgC.qr(A, c(1,0,0), check=FALSE))
})

## Solving sparse least squares:

X <- rbind(A, Diagonal(3)) # design matrix X (for L.S.)
Xt <- t(X)                 # *transposed*  X (for L.S.)
(y <- drop(crossprod(Xt, 1:3)) + c(-1,1)/1000) # small rand.err.
str(solveCh <- .solve.dgC.chol(Xt, y, check=FALSE)) # Xt *is* dgC..
stopifnot(exprs = {
    all.equal(solveCh$coef, 1:3, tol = 1e-3)# rel.err ~ 1e-4
    all.equal(solveCh$coef, drop(solve(tcrossprod(Xt), Xt %*% y)))
    all.equal(solveCh$coef, .solve.dgC.qr(X, y, check=FALSE))
})



cleanEx()
nameEx("forceSymmetric")
### * forceSymmetric

flush(stderr()); flush(stdout())

### Name: forceSymmetric
### Title: Force a Matrix to 'symmetricMatrix' Without Symmetry Checks
### Aliases: forceSymmetric forceSymmetric,CsparseMatrix,character-method
###   forceSymmetric,CsparseMatrix,missing-method
###   forceSymmetric,RsparseMatrix,character-method
###   forceSymmetric,RsparseMatrix,missing-method
###   forceSymmetric,TsparseMatrix,character-method
###   forceSymmetric,TsparseMatrix,missing-method
###   forceSymmetric,diagonalMatrix,character-method
###   forceSymmetric,diagonalMatrix,missing-method
###   forceSymmetric,indMatrix,character-method
###   forceSymmetric,indMatrix,missing-method
###   forceSymmetric,matrix,character-method
###   forceSymmetric,matrix,missing-method
###   forceSymmetric,packedMatrix,character-method
###   forceSymmetric,packedMatrix,missing-method
###   forceSymmetric,unpackedMatrix,character-method
###   forceSymmetric,unpackedMatrix,missing-method
### Keywords: array

### ** Examples

 ## Hilbert matrix
 i <- 1:6
 h6 <- 1/outer(i - 1L, i, "+")
 sd <- sqrt(diag(h6))
 hh <- t(h6/sd)/sd # theoretically symmetric
 isSymmetric(hh, tol=0) # FALSE; hence
 try( as(hh, "symmetricMatrix") ) # fails, but this works fine:
 H6 <- forceSymmetric(hh)

 ## result can be pretty surprising:
 (M <- Matrix(1:36, 6))
 forceSymmetric(M) # symmetric, hence very different in lower triangle
 (tm <- tril(M))
 forceSymmetric(tm)



cleanEx()
nameEx("formatSparseM")
### * formatSparseM

flush(stderr()); flush(stdout())

### Name: formatSparseM
### Title: Formatting Sparse Numeric Matrices Utilities
### Aliases: formatSparseM .formatSparseSimple
### Keywords: utilities print

### ** Examples

m <- suppressWarnings(matrix(c(0, 3.2, 0,0, 11,0,0,0,0,-7,0), 4,9))
fm <- formatSparseM(m)
noquote(fm)
## nice, but this is nicer {with "units" vertically aligned}:
print(fm, quote=FALSE, right=TRUE)
## and "the same" as :
Matrix(m)

## align = "right" is cheaper -->  the "." are not aligned:
noquote(f2 <- formatSparseM(m,align="r"))
stopifnot(f2 == fm   |   m == 0, dim(f2) == dim(m),
         (f2 == ".") == (m == 0))



cleanEx()
nameEx("graph2T")
### * graph2T

flush(stderr()); flush(stdout())

### Name: graph-sparseMatrix
### Title: Conversions "graph" <-> (sparse) Matrix
### Aliases: graph2T T2graph coerce,Matrix,graph-method
###   coerce,Matrix,graphNEL-method coerce,TsparseMatrix,graphNEL-method
###   coerce,graph,CsparseMatrix-method coerce,graph,Matrix-method
###   coerce,graph,RsparseMatrix-method coerce,graph,TsparseMatrix-method
###   coerce,graph,sparseMatrix-method coerce,graphAM,TsparseMatrix-method
###   coerce,graphNEL,TsparseMatrix-method
### Keywords: graph utilities

### ** Examples

if(isTRUE(try(require(graph)))) { ## super careful .. for "checking reasons"
  n4 <- LETTERS[1:4]; dns <- list(n4,n4)
  show(a1 <- sparseMatrix(i= c(1:4),   j=c(2:4,1),   x = 2,    dimnames=dns))
  show(g1 <- as(a1, "graph")) # directed
  unlist(edgeWeights(g1)) # all '2'

  show(a2 <- sparseMatrix(i= c(1:4,4), j=c(2:4,1:2), x = TRUE, dimnames=dns))
  show(g2 <- as(a2, "graph")) # directed
  # now if you want it undirected:
  show(g3  <- T2graph(as(a2,"TsparseMatrix"), edgemode="undirected"))
  show(m3 <- as(g3,"Matrix"))
  show( graph2T(g3) ) # a "pattern Matrix" (nsTMatrix)
## Don't show: 
  stopifnot(
   identical(as(g3,"Matrix"), as(as(a2 + t(a2), "nMatrix"),"symmetricMatrix"))
  ,
   identical(tg3 <- graph2T(g3), graph2T(g3, use.weights=FALSE))
  ,
   identical(as(m3,"TsparseMatrix"), uniqTsparse(tg3))
  )
## End(Don't show)
  a. <- sparseMatrix(i= 4:1, j=1:4, dimnames=list(n4,n4), giveC=FALSE) # no 'x'
  show(a.) # "ngTMatrix"
  show(g. <- as(a., "graph"))
## Don't show: 
  stopifnot(edgemode(g.) == "undirected", numEdges(g.) == 2,
            all.equal(as(g., "TsparseMatrix"),
                      as(a., "symmetricMatrix"))
)
## End(Don't show)
}



cleanEx()
nameEx("image-methods")
### * image-methods

flush(stderr()); flush(stdout())

### Name: image-methods
### Title: Methods for image() in Package 'Matrix'
### Aliases: image-methods image,ANY-method image,CHMfactor-method
###   image,Matrix-method image,dgTMatrix-method
### Keywords: methods hplot

### ** Examples

showMethods(image)
## If you want to see all the methods' implementations:
showMethods(image, incl=TRUE, inherit=FALSE)
## Don't show: 
## warnings should not happen here, notably when print(<trellis>)
op <- options(warn = 2)
## End(Don't show)
data(CAex)
image(CAex, main = "image(CAex)") -> imgC; imgC
stopifnot(!is.null(leg <- imgC$legend), is.list(leg$right)) # failed for 2 days ..
image(CAex, useAbs=TRUE, main = "image(CAex, useAbs=TRUE)")

cCA <- Cholesky(crossprod(CAex), Imult = .01)
## See  ?print.trellis --- place two image() plots side by side:
print(image(cCA, main="Cholesky(crossprod(CAex), Imult = .01)"),
      split=c(x=1,y=1,nx=2, ny=1), more=TRUE)
print(image(cCA, useAbs=TRUE),
      split=c(x=2,y=1,nx=2,ny=1))

data(USCounties)
image(USCounties)# huge
image(sign(USCounties))## just the pattern
    # how the result looks, may depend heavily on
    # the device, screen resolution, antialiasing etc
    # e.g. x11(type="Xlib") may show very differently than cairo-based

## Drawing borders around each rectangle;
    # again, viewing depends very much on the device:
image(USCounties[1:400,1:200], lwd=.1)
## Using (xlim,ylim) has advantage : matrix dimension and (col/row) indices:
image(USCounties, c(1,200), c(1,400), lwd=.1)
image(USCounties, c(1,300), c(1,200), lwd=.5 )
image(USCounties, c(1,300), c(1,200), lwd=.01)
## These 3 are all equivalent :
(I1 <- image(USCounties, c(1,100), c(1,100), useAbs=FALSE))
 I2 <- image(USCounties, c(1,100), c(1,100), useAbs=FALSE,        border.col=NA)
 I3 <- image(USCounties, c(1,100), c(1,100), useAbs=FALSE, lwd=2, border.col=NA)
stopifnot(all.equal(I1, I2, check.environment=FALSE),
          all.equal(I2, I3, check.environment=FALSE))
## using an opaque border color
image(USCounties, c(1,100), c(1,100), useAbs=FALSE, lwd=3, border.col = adjustcolor("skyblue", 1/2))
## Don't show: 
options(op)
## End(Don't show)
if(interactive() || nzchar(Sys.getenv("R_MATRIX_CHECK_EXTRA"))) {
## Using raster graphics: For PDF this would give a 77 MB file,
## however, for such a large matrix, this is typically considerably
## *slower* (than vector graphics rectangles) in most cases :
if(doPNG <- !dev.interactive())
   png("image-USCounties-raster.png", width=3200, height=3200)
image(USCounties, useRaster = TRUE) # should not suffer from anti-aliasing
if(doPNG)
   dev.off()
   ## and now look at the *.png image in a viewer you can easily zoom in and out
}#only if(doExtras)



cleanEx()
nameEx("indMatrix-class")
### * indMatrix-class

flush(stderr()); flush(stdout())

### Name: indMatrix-class
### Title: Index Matrices
### Aliases: indMatrix-class Summary,indMatrix-method
###   as.logical,indMatrix-method as.numeric,indMatrix-method
###   as.vector,indMatrix-method coerce,indMatrix,CsparseMatrix-method
###   coerce,indMatrix,RsparseMatrix-method
###   coerce,indMatrix,TsparseMatrix-method coerce,indMatrix,dMatrix-method
###   coerce,indMatrix,ddenseMatrix-method
###   coerce,indMatrix,denseMatrix-method
###   coerce,indMatrix,diagonalMatrix-method
###   coerce,indMatrix,dsparseMatrix-method
###   coerce,indMatrix,generalMatrix-method coerce,indMatrix,lMatrix-method
###   coerce,indMatrix,ldenseMatrix-method
###   coerce,indMatrix,lsparseMatrix-method coerce,indMatrix,matrix-method
###   coerce,indMatrix,nMatrix-method coerce,indMatrix,ndenseMatrix-method
###   coerce,indMatrix,nsparseMatrix-method coerce,indMatrix,pMatrix-method
###   coerce,indMatrix,packedMatrix-method
###   coerce,indMatrix,unpackedMatrix-method coerce,indMatrix,vector-method
###   coerce,integer,indMatrix-method coerce,list,indMatrix-method
###   coerce,matrix,indMatrix-method coerce,numeric,indMatrix-method
###   diag,indMatrix-method diag<-,indMatrix-method
###   rbind2,indMatrix,indMatrix-method t,indMatrix-method
### Keywords: classes

### ** Examples

p1 <- as(c(2,3,1), "pMatrix")
(sm1 <- as(rep(c(2,3,1), e=3), "indMatrix"))
stopifnot(all(sm1 == p1[rep(1:3, each=3),]))

## row-indexing of a <pMatrix> turns it into an <indMatrix>:
class(p1[rep(1:3, each=3),])

set.seed(12) # so we know '10' is in sample
## random index matrix for 30 observations and 10 unique values:
(s10 <- as(sample(10, 30, replace=TRUE),"indMatrix"))

## Sample rows of a numeric matrix :
(mm <- matrix(1:10, nrow=10, ncol=3))
s10 %*% mm

set.seed(27)
IM1 <- as(sample(1:20, 100, replace=TRUE), "indMatrix")
IM2 <- as(sample(1:18, 100, replace=TRUE), "indMatrix")
(c12 <- crossprod(IM1,IM2))
## same as cross-tabulation of the two index vectors:
stopifnot(all(c12 - unclass(table(IM1@perm, IM2@perm)) == 0))

# 3 observations, 4 implied values, first does not occur in sample:
as(2:4, "indMatrix")
# 3 observations, 5 values, first and last do not occur in sample:
as(list(2:4, 5), "indMatrix")

as(sm1, "nMatrix")
s10[1:7, 1:4] # gives an "ngTMatrix" (most economic!)
s10[1:4, ]  # preserves "indMatrix"-class

I1 <- as(c(5:1,6:4,7:3), "indMatrix")
I2 <- as(7:1, "pMatrix")
(I12 <- rbind(I1, I2))
stopifnot(is(I12, "indMatrix"),
          identical(I12, rbind(I1, I2)),
	  colSums(I12) == c(2L,2:4,4:2))



cleanEx()
nameEx("index-class")
### * index-class

flush(stderr()); flush(stdout())

### Name: index-class
### Title: Virtual Class "index" - Simple Class for Matrix Indices
### Aliases: index-class
### Keywords: classes

### ** Examples

showClass("index")



cleanEx()
nameEx("invPerm")
### * invPerm

flush(stderr()); flush(stdout())

### Name: invPerm
### Title: Inverse Permutation Vector
### Aliases: invPerm
### Keywords: arithmetic

### ** Examples

  p <- sample(10) # a random permutation vector
  ip <- invPerm(p)
  p[ip] # == 1:10
  ## they are indeed inverse of each other:
  stopifnot(
    identical(p[ip], 1:10),
    identical(ip[p], 1:10),
    identical(invPerm(ip), p)
  )
## Don't show: 
 p3 <- c(3, 1:2) # ('double' instead of integer)
 stopifnot(identical(invPerm(p3), c(2:3, 1L)))
## End(Don't show)



cleanEx()
nameEx("is.na-methods")
### * is.na-methods

flush(stderr()); flush(stdout())

### Name: is.na-methods
### Title: is.na(), is.finite() Methods for 'Matrix' Objects
### Aliases: is.na-methods is.nan-methods is.finite-methods
###   is.infinite-methods anyNA-methods is.na,abIndex-method
###   is.na,dgeMatrix-method is.na,diagonalMatrix-method
###   is.na,dspMatrix-method is.na,dsparseMatrix-method
###   is.na,dsyMatrix-method is.na,dtpMatrix-method is.na,dtrMatrix-method
###   is.na,indMatrix-method is.na,lgeMatrix-method is.na,lspMatrix-method
###   is.na,lsparseMatrix-method is.na,lsyMatrix-method
###   is.na,ltpMatrix-method is.na,ltrMatrix-method is.na,nMatrix-method
###   is.na,nsparseVector-method is.na,sparseVector-method
###   is.nan,ddiMatrix-method is.nan,dgeMatrix-method
###   is.nan,dspMatrix-method is.nan,dsparseMatrix-method
###   is.nan,dsyMatrix-method is.nan,dtpMatrix-method
###   is.nan,dtrMatrix-method is.nan,indMatrix-method is.nan,lMatrix-method
###   is.nan,nMatrix-method is.nan,nsparseVector-method
###   is.nan,sparseVector-method is.finite,abIndex-method
###   is.finite,dgeMatrix-method is.finite,diagonalMatrix-method
###   is.finite,dspMatrix-method is.finite,dsparseMatrix-method
###   is.finite,dsyMatrix-method is.finite,dtpMatrix-method
###   is.finite,dtrMatrix-method is.finite,indMatrix-method
###   is.finite,lgeMatrix-method is.finite,lspMatrix-method
###   is.finite,lsparseMatrix-method is.finite,lsyMatrix-method
###   is.finite,ltpMatrix-method is.finite,ltrMatrix-method
###   is.finite,nMatrix-method is.finite,nsparseVector-method
###   is.finite,sparseVector-method is.infinite,abIndex-method
###   is.infinite,ddiMatrix-method is.infinite,dgeMatrix-method
###   is.infinite,dspMatrix-method is.infinite,dsparseMatrix-method
###   is.infinite,dsyMatrix-method is.infinite,dtpMatrix-method
###   is.infinite,dtrMatrix-method is.infinite,indMatrix-method
###   is.infinite,lMatrix-method is.infinite,nMatrix-method
###   is.infinite,nsparseVector-method is.infinite,sparseVector-method
###   anyNA,ddenseMatrix-method anyNA,diagonalMatrix-method
###   anyNA,dsparseMatrix-method anyNA,indMatrix-method
###   anyNA,ldenseMatrix-method anyNA,lsparseMatrix-method
###   anyNA,nMatrix-method anyNA,nsparseVector-method
###   anyNA,sparseVector-method
### Keywords: methods

### ** Examples

(M <- Matrix(1:6, nrow = 4, ncol = 3,
             dimnames = list(letters[1:4], LETTERS[1:3])))
stopifnot(!anyNA(M), !any(is.na(M)))

M[2:3, 2] <- NA
(inM <- is.na(M))
stopifnot(anyNA(M), sum(inM) == 2)

(A <- spMatrix(nrow = 10, ncol = 20,
               i = c(1, 3:8), j = c(2, 9, 6:10), x = 7 * (1:7)))
stopifnot(!anyNA(A), !any(is.na(A)))

A[2, 3] <- A[1, 2] <- A[5, 5:9] <- NA
(inA <- is.na(A))
stopifnot(anyNA(A), sum(inA) == 1 + 1 + 5)



cleanEx()
nameEx("is.null.DN")
### * is.null.DN

flush(stderr()); flush(stdout())

### Name: is.null.DN
### Title: Are the Dimnames 'dn' NULL-like ?
### Aliases: is.null.DN
### Keywords: utilities

### ** Examples

m <- matrix(round(100 * rnorm(6)), 2,3); m1 <- m2 <- m3 <- m4 <- m
dimnames(m1) <- list(NULL, NULL)
dimnames(m2) <- list(NULL, character())
dimnames(m3) <- rev(dimnames(m2))
dimnames(m4) <- rep(list(character()),2)

m4 ## prints absolutely identically to  m

stopifnot(m == m1, m1 == m2, m2 == m3, m3 == m4,
	  identical(capture.output(m) -> cm,
		    capture.output(m1)),
	  identical(cm, capture.output(m2)),
	  identical(cm, capture.output(m3)),
	  identical(cm, capture.output(m4)))

hasNoDimnames <- function(.) is.null.DN(dimnames(.))

stopifnot(exprs = {
  hasNoDimnames(m)
  hasNoDimnames(m1); hasNoDimnames(m2)
  hasNoDimnames(m3); hasNoDimnames(m4)
  hasNoDimnames(Matrix(m) -> M)
  hasNoDimnames(as(M, "sparseMatrix"))
})




cleanEx()
nameEx("isSymmetric-methods")
### * isSymmetric-methods

flush(stderr()); flush(stdout())

### Name: isSymmetric-methods
### Title: Methods for Function 'isSymmetric' in Package 'Matrix'
### Aliases: isSymmetric-methods isSymmetric,diagonalMatrix-method
###   isSymmetric,indMatrix-method isSymmetric,symmetricMatrix-method
###   isSymmetric,triangularMatrix-method isSymmetric,dgCMatrix-method
###   isSymmetric,dgRMatrix-method isSymmetric,dgTMatrix-method
###   isSymmetric,dgeMatrix-method isSymmetric,lgCMatrix-method
###   isSymmetric,lgRMatrix-method isSymmetric,lgTMatrix-method
###   isSymmetric,lgeMatrix-method isSymmetric,ngCMatrix-method
###   isSymmetric,ngRMatrix-method isSymmetric,ngTMatrix-method
###   isSymmetric,ngeMatrix-method isSymmetric,dtCMatrix-method
###   isSymmetric,dtRMatrix-method isSymmetric,dtTMatrix-method
###   isSymmetric,dtpMatrix-method isSymmetric,dtrMatrix-method
### Keywords: methods

### ** Examples

isSymmetric(Diagonal(4)) # TRUE of course
M <- Matrix(c(1,2,2,1), 2,2)
isSymmetric(M) # TRUE (*and* of formal class "dsyMatrix")
isSymmetric(as(M, "generalMatrix")) # still symmetric, even if not "formally"
isSymmetric(triu(M)) # FALSE

## Look at implementations:
showMethods("isSymmetric", includeDefs = TRUE) # includes S3 generic from base



cleanEx()
nameEx("isTriangular")
### * isTriangular

flush(stderr()); flush(stdout())

### Name: isTriangular
### Title: Test whether a Matrix is Triangular or Diagonal
### Aliases: isTriangular isTriangular-methods isDiagonal
###   isDiagonal-methods isTriangular,diagonalMatrix-method
###   isTriangular,indMatrix-method isTriangular,matrix-method
###   isTriangular,symmetricMatrix-method
###   isTriangular,triangularMatrix-method isDiagonal,CsparseMatrix-method
###   isDiagonal,RsparseMatrix-method isDiagonal,TsparseMatrix-method
###   isDiagonal,diagonalMatrix-method isDiagonal,indMatrix-method
###   isDiagonal,matrix-method isDiagonal,packedMatrix-method
###   isDiagonal,unpackedMatrix-method isTriangular,dgCMatrix-method
###   isTriangular,dgRMatrix-method isTriangular,dgTMatrix-method
###   isTriangular,dgeMatrix-method isTriangular,lgCMatrix-method
###   isTriangular,lgRMatrix-method isTriangular,lgTMatrix-method
###   isTriangular,lgeMatrix-method isTriangular,ngCMatrix-method
###   isTriangular,ngRMatrix-method isTriangular,ngTMatrix-method
###   isTriangular,ngeMatrix-method
### Keywords: methods

### ** Examples

isTriangular(Diagonal(4))
## is TRUE: a diagonal matrix is also (both upper and lower) triangular
(M <- Matrix(c(1,2,0,1), 2,2))
isTriangular(M) # TRUE (*and* of formal class "dtrMatrix")
isTriangular(as(M, "generalMatrix")) # still triangular, even if not "formally"
isTriangular(crossprod(M)) # FALSE

isDiagonal(matrix(c(2,0,0,1), 2,2)) # TRUE

## Look at implementations:
showMethods("isTriangular", includeDefs = TRUE)
showMethods("isDiagonal", includeDefs = TRUE)



cleanEx()
nameEx("kronecker-methods")
### * kronecker-methods

flush(stderr()); flush(stdout())

### Name: kronecker-methods
### Title: Methods for Function 'kronecker()' in Package 'Matrix'
### Aliases: kronecker-methods kronecker,CsparseMatrix,CsparseMatrix-method
###   kronecker,CsparseMatrix,Matrix-method
###   kronecker,CsparseMatrix,diagonalMatrix-method
###   kronecker,Matrix,matrix-method kronecker,Matrix,vector-method
###   kronecker,RsparseMatrix,Matrix-method
###   kronecker,RsparseMatrix,RsparseMatrix-method
###   kronecker,RsparseMatrix,diagonalMatrix-method
###   kronecker,TsparseMatrix,Matrix-method
###   kronecker,TsparseMatrix,TsparseMatrix-method
###   kronecker,TsparseMatrix,diagonalMatrix-method
###   kronecker,denseMatrix,Matrix-method
###   kronecker,denseMatrix,denseMatrix-method
###   kronecker,diagonalMatrix,CsparseMatrix-method
###   kronecker,diagonalMatrix,Matrix-method
###   kronecker,diagonalMatrix,RsparseMatrix-method
###   kronecker,diagonalMatrix,TsparseMatrix-method
###   kronecker,diagonalMatrix,diagonalMatrix-method
###   kronecker,diagonalMatrix,indMatrix-method
###   kronecker,indMatrix,Matrix-method
###   kronecker,indMatrix,diagonalMatrix-method
###   kronecker,indMatrix,indMatrix-method kronecker,matrix,Matrix-method
###   kronecker,vector,Matrix-method
### Keywords: methods array

### ** Examples

(t1 <- spMatrix(5,4, x= c(3,2,-7,11), i= 1:4, j=4:1)) #  5 x  4
(t2 <- kronecker(Diagonal(3, 2:4), t1))               # 15 x 12

## should also work with special-cased logical matrices
l3 <- upper.tri(matrix(,3,3))
M <- Matrix(l3)
(N <- as(M, "nsparseMatrix")) # "ntCMatrix" (upper triangular)
N2 <- as(N, "generalMatrix")  # (lost "t"riangularity)
MM <- kronecker(M,M)
NN <- kronecker(N,N) # "dtTMatrix" i.e. did keep
NN2 <- kronecker(N2,N2)
stopifnot(identical(NN,MM),
          is(NN2, "sparseMatrix"), all(NN2 == NN),
          is(NN, "triangularMatrix"))



cleanEx()
nameEx("ldenseMatrix-class")
### * ldenseMatrix-class

flush(stderr()); flush(stdout())

### Name: ldenseMatrix-class
### Title: Virtual Class "ldenseMatrix" of Dense Logical Matrices
### Aliases: ldenseMatrix-class &,ldenseMatrix,ddiMatrix-method
###   &,ldenseMatrix,ldiMatrix-method *,ldenseMatrix,ddiMatrix-method
###   *,ldenseMatrix,ldiMatrix-method
###   Logic,ldenseMatrix,lsparseMatrix-method
###   Ops,ldenseMatrix,ldenseMatrix-method Summary,ldenseMatrix-method
###   ^,ldenseMatrix,ddiMatrix-method ^,ldenseMatrix,ldiMatrix-method
###   coerce,matrix,ldenseMatrix-method coerce,numLike,ldenseMatrix-method
###   which,ldenseMatrix-method
### Keywords: classes

### ** Examples

showClass("ldenseMatrix")

as(diag(3) > 0, "ldenseMatrix")



cleanEx()
nameEx("ldiMatrix-class")
### * ldiMatrix-class

flush(stderr()); flush(stdout())

### Name: ldiMatrix-class
### Title: Class "ldiMatrix" of Diagonal Logical Matrices
### Aliases: ldiMatrix-class !,ldiMatrix-method %%,ldiMatrix,Matrix-method
###   %%,ldiMatrix,ddenseMatrix-method %%,ldiMatrix,ldenseMatrix-method
###   %%,ldiMatrix,ndenseMatrix-method %/%,ldiMatrix,Matrix-method
###   %/%,ldiMatrix,ddenseMatrix-method %/%,ldiMatrix,ldenseMatrix-method
###   %/%,ldiMatrix,ndenseMatrix-method &,ldiMatrix,Matrix-method
###   &,ldiMatrix,ddenseMatrix-method &,ldiMatrix,ldenseMatrix-method
###   &,ldiMatrix,ndenseMatrix-method *,ldiMatrix,Matrix-method
###   *,ldiMatrix,ddenseMatrix-method *,ldiMatrix,ldenseMatrix-method
###   *,ldiMatrix,ndenseMatrix-method -,ldiMatrix,missing-method
###   /,ldiMatrix,Matrix-method /,ldiMatrix,ddenseMatrix-method
###   /,ldiMatrix,ldenseMatrix-method /,ldiMatrix,ndenseMatrix-method
###   Arith,ldiMatrix,logical-method Arith,ldiMatrix,numeric-method
###   Arith,logical,ldiMatrix-method Arith,numeric,ldiMatrix-method
###   Ops,ANY,ldiMatrix-method Ops,ldiMatrix,ANY-method
###   Ops,ldiMatrix,Matrix-method Ops,ldiMatrix,dMatrix-method
###   Ops,ldiMatrix,ddiMatrix-method Ops,ldiMatrix,ldiMatrix-method
###   Ops,ldiMatrix,logical-method Ops,ldiMatrix,numeric-method
###   Ops,ldiMatrix,sparseMatrix-method Summary,ldiMatrix-method
###   as.logical,ldiMatrix-method cbind2,ldiMatrix,atomicVector-method
###   cbind2,ldiMatrix,matrix-method cbind2,matrix,ldiMatrix-method
###   prod,ldiMatrix-method rbind2,ldiMatrix,atomicVector-method
###   rbind2,ldiMatrix,matrix-method rbind2,matrix,ldiMatrix-method
###   sum,ldiMatrix-method which,ldiMatrix-method
### Keywords: classes

### ** Examples

(lM <- Diagonal(x = c(TRUE,FALSE,FALSE)))
str(lM)#> gory details (slots)

crossprod(lM) # numeric
(nM <- as(lM, "nMatrix"))# -> sparse (not formally ``diagonal'')
crossprod(nM) # logical sparse



cleanEx()
nameEx("lgeMatrix-class")
### * lgeMatrix-class

flush(stderr()); flush(stdout())

### Name: lgeMatrix-class
### Title: Class "lgeMatrix" of General Dense Logical Matrices
### Aliases: lgeMatrix-class !,lgeMatrix-method
###   Arith,lgeMatrix,lgeMatrix-method Compare,lgeMatrix,lgeMatrix-method
###   Logic,lgeMatrix,lgeMatrix-method as.vector,lgeMatrix-method
###   coerce,lgeMatrix,matrix-method coerce,lgeMatrix,vector-method
### Keywords: classes

### ** Examples

showClass("lgeMatrix")
str(new("lgeMatrix"))
set.seed(1)
(lM <- Matrix(matrix(rnorm(28), 4,7) > 0))# a simple random lgeMatrix
set.seed(11)
(lC <- Matrix(matrix(rnorm(28), 4,7) > 0))# a simple random lgCMatrix
as(lM, "CsparseMatrix")



cleanEx()
nameEx("lsparseMatrix-classes")
### * lsparseMatrix-classes

flush(stderr()); flush(stdout())

### Name: lsparseMatrix-classes
### Title: Sparse logical matrices
### Aliases: lsparseMatrix-class lgCMatrix-class lgRMatrix-class
###   lgTMatrix-class ltCMatrix-class ltRMatrix-class ltTMatrix-class
###   lsCMatrix-class lsRMatrix-class lsTMatrix-class
###   !,lsparseMatrix-method Arith,lsparseMatrix,Matrix-method
###   Logic,lsparseMatrix,ldenseMatrix-method
###   Logic,lsparseMatrix,lsparseMatrix-method
###   Ops,lsparseMatrix,lsparseMatrix-method
###   Ops,lsparseMatrix,nsparseMatrix-method
###   as.logical,lsparseMatrix-method as.numeric,lsparseMatrix-method
###   coerce,lsparseMatrix,dMatrix-method
###   coerce,lsparseMatrix,dsparseMatrix-method
###   coerce,lsparseMatrix,nMatrix-method
###   coerce,lsparseMatrix,nsparseMatrix-method
###   coerce,matrix,lsparseMatrix-method
###   coerce,numLike,lsparseMatrix-method which,lsparseMatrix-method
###   Arith,lgCMatrix,lgCMatrix-method Logic,lgCMatrix,lgCMatrix-method
###   Arith,lgTMatrix,lgTMatrix-method Logic,lgTMatrix,lgTMatrix-method
###   which,lgTMatrix-method Logic,ltCMatrix,ltCMatrix-method
###   which,ltTMatrix-method Logic,lsCMatrix,lsCMatrix-method
###   coerce,lsCMatrix,RsparseMatrix-method
###   coerce,lsRMatrix,CsparseMatrix-method which,lsTMatrix-method
### Keywords: classes algebra

### ** Examples

(m <- Matrix(c(0,0,2:0), 3,5, dimnames=list(LETTERS[1:3],NULL)))
(lm <- (m > 1)) # lgC
!lm     # no longer sparse
stopifnot(is(lm,"lsparseMatrix"),
          identical(!lm, m <= 1))

data(KNex)
str(mmG.1 <- (KNex $ mm) > 0.1)# "lgC..."
table(mmG.1@x)# however with many ``non-structural zeros''
## from logical to nz_pattern -- okay when there are no NA's :
nmG.1 <- as(mmG.1, "nMatrix") # <<< has "TRUE" also where mmG.1 had FALSE
## from logical to "double"
dmG.1 <- as(mmG.1, "dMatrix") # has '0' and back:
lmG.1 <- as(dmG.1, "lMatrix")
stopifnot(identical(nmG.1, as((KNex $ mm) != 0,"nMatrix")),
          validObject(lmG.1),
          identical(lmG.1, mmG.1))

class(xnx <- crossprod(nmG.1))# "nsC.."
class(xlx <- crossprod(mmG.1))# "dsC.." : numeric
is0 <- (xlx == 0)
mean(as.vector(is0))# 99.3% zeros: quite sparse, but
table(xlx@x == 0)# more than half of the entries are (non-structural!) 0
stopifnot(isSymmetric(xlx), isSymmetric(xnx),
          ## compare xnx and xlx : have the *same* non-structural 0s :
          sapply(slotNames(xnx),
                 function(n) identical(slot(xnx, n), slot(xlx, n))))



cleanEx()
nameEx("lsyMatrix-class")
### * lsyMatrix-class

flush(stderr()); flush(stdout())

### Name: lsyMatrix-class
### Title: Symmetric Dense Logical Matrices
### Aliases: lsyMatrix-class lspMatrix-class !,lsyMatrix-method
###   !,lspMatrix-method
### Keywords: classes

### ** Examples

(M2 <- Matrix(c(TRUE, NA, FALSE, FALSE), 2, 2)) # logical dense (ltr)
str(M2)
# can
(sM <- M2 | t(M2)) # "lge"
as(sM, "symmetricMatrix")
str(sM <- as(sM, "packedMatrix")) # packed symmetric



cleanEx()
nameEx("ltrMatrix-class")
### * ltrMatrix-class

flush(stderr()); flush(stdout())

### Name: ltrMatrix-class
### Title: Triangular Dense Logical Matrices
### Aliases: ltrMatrix-class ltpMatrix-class !,ltrMatrix-method
###   !,ltpMatrix-method
### Keywords: classes

### ** Examples

showClass("ltrMatrix")

str(new("ltpMatrix"))
(lutr <- as(upper.tri(matrix(, 4, 4)), "ldenseMatrix"))
str(lutp <- pack(lutr)) # packed matrix: only 10 = 4*(4+1)/2 entries
!lutp # the logical negation (is *not* logical triangular !)
## but this one is:
stopifnot(all.equal(lutp, pack(!!lutp)))



cleanEx()
nameEx("lu")
### * lu

flush(stderr()); flush(stdout())

### Name: lu
### Title: (Generalized) Triangular Decomposition of a Matrix
### Aliases: lu lu,denseMatrix-method lu,diagonalMatrix-method
###   lu,dgCMatrix-method lu,dgRMatrix-method lu,dgTMatrix-method
###   lu,dgeMatrix-method lu,dsCMatrix-method lu,dsRMatrix-method
###   lu,dsTMatrix-method lu,dspMatrix-method lu,dsyMatrix-method
###   lu,dtCMatrix-method lu,dtRMatrix-method lu,dtTMatrix-method
###   lu,dtpMatrix-method lu,dtrMatrix-method lu,matrix-method
###   lu,sparseMatrix-method
### Keywords: array algebra

### ** Examples


##--- Dense  -------------------------
x <- Matrix(rnorm(9), 3, 3)
lu(x)
dim(x2 <- round(10 * x[,-3]))# non-square
expand(lu2 <- lu(x2))

##--- Sparse (see more in ?"sparseLU-class")----- % ./sparseLU-class.Rd

pm <- as(readMM(system.file("external/pores_1.mtx",
                            package = "Matrix")),
         "CsparseMatrix")
str(pmLU <- lu(pm))		# p is a 0-based permutation of the rows
                                # q is a 0-based permutation of the columns
## permute rows and columns of original matrix
ppm <- pm[pmLU@p + 1L, pmLU@q + 1L]
pLU <- drop0(pmLU@L %*% pmLU@U) # L %*% U -- dropping extra zeros
## equal up to "rounding"
ppm[1:14, 1:5]
pLU[1:14, 1:5]



cleanEx()
nameEx("mat2triplet")
### * mat2triplet

flush(stderr()); flush(stdout())

### Name: mat2triplet
### Title: Map Matrix to its Triplet Representation
### Aliases: mat2triplet
### Keywords: classes manip utilities

### ** Examples

if(FALSE) ## The function is defined (don't redefine here!), simply as
mat2triplet <- function(x, uniqT = FALSE) {
    T <- as(x, "TsparseMatrix")
    if(uniqT && anyDuplicatedT(T)) T <- .uniqTsparse(T)
    if(is(T, "nsparseMatrix"))
         list(i = T@i + 1L, j = T@j + 1L)
    else list(i = T@i + 1L, j = T@j + 1L, x = T@x)
}

i <- c(1,3:8); j <- c(2,9,6:10); x <- 7 * (1:7)
(Ax <- sparseMatrix(i, j, x = x)) ##  8 x 10 "dgCMatrix"
str(trA <- mat2triplet(Ax))
stopifnot(i == sort(trA$i),  sort(j) == trA$j,  x == sort(trA$x))

D <- Diagonal(x=4:2)
summary(D)
str(mat2triplet(D))



cleanEx()
nameEx("matrix-products")
### * matrix-products

flush(stderr()); flush(stdout())

### Name: matrix-products
### Title: Matrix (Cross) Products (of Transpose)
### Aliases: %*%-methods crossprod-methods tcrossprod-methods %*% crossprod
###   tcrossprod %*%,ANY,Matrix-method %*%,ANY,TsparseMatrix-method
###   %*%,CsparseMatrix,CsparseMatrix-method
###   %*%,CsparseMatrix,ddenseMatrix-method
###   %*%,CsparseMatrix,diagonalMatrix-method
###   %*%,CsparseMatrix,matrix-method %*%,CsparseMatrix,numLike-method
###   %*%,Matrix,ANY-method %*%,Matrix,TsparseMatrix-method
###   %*%,Matrix,indMatrix-method %*%,Matrix,matrix-method
###   %*%,Matrix,numLike-method %*%,Matrix,pMatrix-method
###   %*%,RsparseMatrix,diagonalMatrix-method
###   %*%,RsparseMatrix,mMatrix-method %*%,TsparseMatrix,ANY-method
###   %*%,TsparseMatrix,Matrix-method
###   %*%,TsparseMatrix,TsparseMatrix-method
###   %*%,TsparseMatrix,diagonalMatrix-method %*%,dMatrix,lMatrix-method
###   %*%,dMatrix,nMatrix-method %*%,ddenseMatrix,CsparseMatrix-method
###   %*%,ddenseMatrix,ddenseMatrix-method
###   %*%,ddenseMatrix,dsyMatrix-method %*%,ddenseMatrix,dtrMatrix-method
###   %*%,ddenseMatrix,ldenseMatrix-method %*%,ddenseMatrix,matrix-method
###   %*%,ddenseMatrix,ndenseMatrix-method
###   %*%,denseMatrix,diagonalMatrix-method %*%,dgeMatrix,dgeMatrix-method
###   %*%,dgeMatrix,dtpMatrix-method %*%,dgeMatrix,matrix-method
###   %*%,diagonalMatrix,CsparseMatrix-method
###   %*%,diagonalMatrix,RsparseMatrix-method
###   %*%,diagonalMatrix,TsparseMatrix-method
###   %*%,diagonalMatrix,denseMatrix-method
###   %*%,diagonalMatrix,diagonalMatrix-method
###   %*%,diagonalMatrix,matrix-method %*%,dspMatrix,ddenseMatrix-method
###   %*%,dspMatrix,matrix-method %*%,dsyMatrix,ddenseMatrix-method
###   %*%,dsyMatrix,dsyMatrix-method %*%,dsyMatrix,matrix-method
###   %*%,dtpMatrix,ddenseMatrix-method %*%,dtpMatrix,matrix-method
###   %*%,dtrMatrix,ddenseMatrix-method %*%,dtrMatrix,dtrMatrix-method
###   %*%,dtrMatrix,matrix-method %*%,indMatrix,Matrix-method
###   %*%,indMatrix,indMatrix-method %*%,indMatrix,matrix-method
###   %*%,indMatrix,pMatrix-method %*%,lMatrix,dMatrix-method
###   %*%,lMatrix,lMatrix-method %*%,lMatrix,nMatrix-method
###   %*%,ldenseMatrix,ddenseMatrix-method
###   %*%,ldenseMatrix,ldenseMatrix-method
###   %*%,ldenseMatrix,lsparseMatrix-method %*%,ldenseMatrix,matrix-method
###   %*%,ldenseMatrix,ndenseMatrix-method
###   %*%,lsparseMatrix,ldenseMatrix-method
###   %*%,lsparseMatrix,lsparseMatrix-method
###   %*%,mMatrix,RsparseMatrix-method %*%,mMatrix,sparseVector-method
###   %*%,matrix,CsparseMatrix-method %*%,matrix,Matrix-method
###   %*%,matrix,ddenseMatrix-method %*%,matrix,dgeMatrix-method
###   %*%,matrix,diagonalMatrix-method %*%,matrix,dsyMatrix-method
###   %*%,matrix,dtpMatrix-method %*%,matrix,dtrMatrix-method
###   %*%,matrix,indMatrix-method %*%,matrix,ldenseMatrix-method
###   %*%,matrix,ndenseMatrix-method %*%,matrix,pMatrix-method
###   %*%,matrix,sparseMatrix-method %*%,nMatrix,dMatrix-method
###   %*%,nMatrix,lMatrix-method %*%,nMatrix,nMatrix-method
###   %*%,ndenseMatrix,ddenseMatrix-method
###   %*%,ndenseMatrix,ldenseMatrix-method %*%,ndenseMatrix,matrix-method
###   %*%,ndenseMatrix,ndenseMatrix-method
###   %*%,ndenseMatrix,nsparseMatrix-method
###   %*%,nsparseMatrix,ndenseMatrix-method
###   %*%,nsparseMatrix,nsparseMatrix-method
###   %*%,numLike,CsparseMatrix-method %*%,numLike,Matrix-method
###   %*%,numLike,sparseVector-method %*%,pMatrix,pMatrix-method
###   %*%,sparseMatrix,matrix-method %*%,sparseVector,mMatrix-method
###   %*%,sparseVector,numLike-method %*%,sparseVector,sparseVector-method
###   crossprod,ANY,ANY-method crossprod,ANY,Matrix-method
###   crossprod,ANY,RsparseMatrix-method crossprod,ANY,TsparseMatrix-method
###   crossprod,CsparseMatrix,CsparseMatrix-method
###   crossprod,CsparseMatrix,ddenseMatrix-method
###   crossprod,CsparseMatrix,diagonalMatrix-method
###   crossprod,CsparseMatrix,matrix-method
###   crossprod,CsparseMatrix,missing-method
###   crossprod,CsparseMatrix,numLike-method crossprod,Matrix,ANY-method
###   crossprod,Matrix,Matrix-method crossprod,Matrix,TsparseMatrix-method
###   crossprod,Matrix,indMatrix-method crossprod,Matrix,matrix-method
###   crossprod,Matrix,missing-method crossprod,Matrix,numLike-method
###   crossprod,Matrix,pMatrix-method crossprod,RsparseMatrix,ANY-method
###   crossprod,RsparseMatrix,diagonalMatrix-method
###   crossprod,RsparseMatrix,mMatrix-method
###   crossprod,TsparseMatrix,ANY-method
###   crossprod,TsparseMatrix,Matrix-method
###   crossprod,TsparseMatrix,TsparseMatrix-method
###   crossprod,TsparseMatrix,diagonalMatrix-method
###   crossprod,TsparseMatrix,missing-method
###   crossprod,ddenseMatrix,CsparseMatrix-method
###   crossprod,ddenseMatrix,ddenseMatrix-method
###   crossprod,ddenseMatrix,dgCMatrix-method
###   crossprod,ddenseMatrix,dsparseMatrix-method
###   crossprod,ddenseMatrix,ldenseMatrix-method
###   crossprod,ddenseMatrix,matrix-method
###   crossprod,ddenseMatrix,missing-method
###   crossprod,ddenseMatrix,ndenseMatrix-method
###   crossprod,denseMatrix,diagonalMatrix-method
###   crossprod,dgCMatrix,dgeMatrix-method
###   crossprod,dgeMatrix,dgeMatrix-method
###   crossprod,dgeMatrix,matrix-method crossprod,dgeMatrix,missing-method
###   crossprod,dgeMatrix,numLike-method
###   crossprod,diagonalMatrix,CsparseMatrix-method
###   crossprod,diagonalMatrix,RsparseMatrix-method
###   crossprod,diagonalMatrix,TsparseMatrix-method
###   crossprod,diagonalMatrix,denseMatrix-method
###   crossprod,diagonalMatrix,diagonalMatrix-method
###   crossprod,diagonalMatrix,matrix-method
###   crossprod,diagonalMatrix,missing-method
###   crossprod,dsparseMatrix,ddenseMatrix-method
###   crossprod,dsparseMatrix,dgeMatrix-method
###   crossprod,dtpMatrix,ddenseMatrix-method
###   crossprod,dtpMatrix,matrix-method
###   crossprod,dtrMatrix,ddenseMatrix-method
###   crossprod,dtrMatrix,dtrMatrix-method
###   crossprod,dtrMatrix,matrix-method crossprod,indMatrix,Matrix-method
###   crossprod,indMatrix,indMatrix-method
###   crossprod,indMatrix,matrix-method crossprod,indMatrix,missing-method
###   crossprod,ldenseMatrix,ddenseMatrix-method
###   crossprod,ldenseMatrix,ldenseMatrix-method
###   crossprod,ldenseMatrix,lsparseMatrix-method
###   crossprod,ldenseMatrix,matrix-method
###   crossprod,ldenseMatrix,missing-method
###   crossprod,ldenseMatrix,ndenseMatrix-method
###   crossprod,lsparseMatrix,ldenseMatrix-method
###   crossprod,lsparseMatrix,lsparseMatrix-method
###   crossprod,mMatrix,RsparseMatrix-method
###   crossprod,mMatrix,sparseVector-method
###   crossprod,matrix,CsparseMatrix-method crossprod,matrix,Matrix-method
###   crossprod,matrix,dgeMatrix-method
###   crossprod,matrix,diagonalMatrix-method
###   crossprod,matrix,dtrMatrix-method crossprod,matrix,indMatrix-method
###   crossprod,matrix,pMatrix-method
###   crossprod,ndenseMatrix,ddenseMatrix-method
###   crossprod,ndenseMatrix,ldenseMatrix-method
###   crossprod,ndenseMatrix,matrix-method
###   crossprod,ndenseMatrix,missing-method
###   crossprod,ndenseMatrix,ndenseMatrix-method
###   crossprod,ndenseMatrix,nsparseMatrix-method
###   crossprod,nsparseMatrix,ndenseMatrix-method
###   crossprod,nsparseMatrix,nsparseMatrix-method
###   crossprod,numLike,CsparseMatrix-method
###   crossprod,numLike,Matrix-method crossprod,numLike,dgeMatrix-method
###   crossprod,numLike,sparseVector-method crossprod,pMatrix,Matrix-method
###   crossprod,pMatrix,indMatrix-method crossprod,pMatrix,matrix-method
###   crossprod,pMatrix,missing-method crossprod,pMatrix,pMatrix-method
###   crossprod,sparseVector,mMatrix-method
###   crossprod,sparseVector,missing-method
###   crossprod,sparseVector,numLike-method
###   crossprod,sparseVector,sparseVector-method
###   crossprod,symmetricMatrix,ANY-method
###   crossprod,symmetricMatrix,Matrix-method
###   crossprod,symmetricMatrix,missing-method tcrossprod,ANY,ANY-method
###   tcrossprod,ANY,Matrix-method tcrossprod,ANY,RsparseMatrix-method
###   tcrossprod,ANY,TsparseMatrix-method
###   tcrossprod,ANY,symmetricMatrix-method
###   tcrossprod,CsparseMatrix,CsparseMatrix-method
###   tcrossprod,CsparseMatrix,ddenseMatrix-method
###   tcrossprod,CsparseMatrix,diagonalMatrix-method
###   tcrossprod,CsparseMatrix,matrix-method
###   tcrossprod,CsparseMatrix,missing-method
###   tcrossprod,CsparseMatrix,numLike-method tcrossprod,Matrix,ANY-method
###   tcrossprod,Matrix,Matrix-method
###   tcrossprod,Matrix,TsparseMatrix-method
###   tcrossprod,Matrix,indMatrix-method tcrossprod,Matrix,matrix-method
###   tcrossprod,Matrix,missing-method tcrossprod,Matrix,numLike-method
###   tcrossprod,Matrix,pMatrix-method
###   tcrossprod,Matrix,symmetricMatrix-method
###   tcrossprod,RsparseMatrix,ANY-method
###   tcrossprod,RsparseMatrix,diagonalMatrix-method
###   tcrossprod,RsparseMatrix,mMatrix-method
###   tcrossprod,TsparseMatrix,ANY-method
###   tcrossprod,TsparseMatrix,Matrix-method
###   tcrossprod,TsparseMatrix,TsparseMatrix-method
###   tcrossprod,TsparseMatrix,diagonalMatrix-method
###   tcrossprod,TsparseMatrix,missing-method
###   tcrossprod,ddenseMatrix,CsparseMatrix-method
###   tcrossprod,ddenseMatrix,ddenseMatrix-method
###   tcrossprod,ddenseMatrix,dsCMatrix-method
###   tcrossprod,ddenseMatrix,dtrMatrix-method
###   tcrossprod,ddenseMatrix,ldenseMatrix-method
###   tcrossprod,ddenseMatrix,lsCMatrix-method
###   tcrossprod,ddenseMatrix,matrix-method
###   tcrossprod,ddenseMatrix,missing-method
###   tcrossprod,ddenseMatrix,ndenseMatrix-method
###   tcrossprod,ddenseMatrix,nsCMatrix-method
###   tcrossprod,denseMatrix,diagonalMatrix-method
###   tcrossprod,dgeMatrix,dgeMatrix-method
###   tcrossprod,dgeMatrix,matrix-method
###   tcrossprod,dgeMatrix,missing-method
###   tcrossprod,dgeMatrix,numLike-method
###   tcrossprod,diagonalMatrix,CsparseMatrix-method
###   tcrossprod,diagonalMatrix,RsparseMatrix-method
###   tcrossprod,diagonalMatrix,TsparseMatrix-method
###   tcrossprod,diagonalMatrix,denseMatrix-method
###   tcrossprod,diagonalMatrix,diagonalMatrix-method
###   tcrossprod,diagonalMatrix,matrix-method
###   tcrossprod,diagonalMatrix,missing-method
###   tcrossprod,dtrMatrix,dtrMatrix-method
###   tcrossprod,indMatrix,Matrix-method
###   tcrossprod,indMatrix,indMatrix-method
###   tcrossprod,indMatrix,matrix-method
###   tcrossprod,indMatrix,missing-method
###   tcrossprod,indMatrix,pMatrix-method
###   tcrossprod,ldenseMatrix,ddenseMatrix-method
###   tcrossprod,ldenseMatrix,ldenseMatrix-method
###   tcrossprod,ldenseMatrix,matrix-method
###   tcrossprod,ldenseMatrix,missing-method
###   tcrossprod,ldenseMatrix,ndenseMatrix-method
###   tcrossprod,mMatrix,RsparseMatrix-method
###   tcrossprod,mMatrix,sparseVector-method
###   tcrossprod,matrix,CsparseMatrix-method
###   tcrossprod,matrix,Matrix-method tcrossprod,matrix,dgeMatrix-method
###   tcrossprod,matrix,diagonalMatrix-method
###   tcrossprod,matrix,dsCMatrix-method tcrossprod,matrix,dtrMatrix-method
###   tcrossprod,matrix,indMatrix-method tcrossprod,matrix,lsCMatrix-method
###   tcrossprod,matrix,nsCMatrix-method tcrossprod,matrix,pMatrix-method
###   tcrossprod,ndenseMatrix,ddenseMatrix-method
###   tcrossprod,ndenseMatrix,ldenseMatrix-method
###   tcrossprod,ndenseMatrix,matrix-method
###   tcrossprod,ndenseMatrix,missing-method
###   tcrossprod,ndenseMatrix,ndenseMatrix-method
###   tcrossprod,numLike,CsparseMatrix-method
###   tcrossprod,numLike,Matrix-method tcrossprod,numLike,dgeMatrix-method
###   tcrossprod,numLike,sparseVector-method
###   tcrossprod,pMatrix,missing-method tcrossprod,pMatrix,pMatrix-method
###   tcrossprod,sparseMatrix,sparseVector-method
###   tcrossprod,sparseVector,mMatrix-method
###   tcrossprod,sparseVector,missing-method
###   tcrossprod,sparseVector,numLike-method
###   tcrossprod,sparseVector,sparseMatrix-method
###   tcrossprod,sparseVector,sparseVector-method
### Keywords: methods algebra

### ** Examples

 ## A random sparse "incidence" matrix :
 m <- matrix(0, 400, 500)
 set.seed(12)
 m[runif(314, 0, length(m))] <- 1
 mm <- as(m, "CsparseMatrix")
 object.size(m) / object.size(mm) # smaller by a factor of > 200

 ## tcrossprod() is very fast:
 system.time(tCmm <- tcrossprod(mm))# 0   (PIII, 933 MHz)
 system.time(cm <- crossprod(t(m))) # 0.16
 system.time(cm. <- tcrossprod(m))  # 0.02

 stopifnot(cm == as(tCmm, "matrix"))

 ## show sparse sub matrix
 tCmm[1:16, 1:30]



cleanEx()
nameEx("nMatrix-class")
### * nMatrix-class

flush(stderr()); flush(stdout())

### Name: nMatrix-class
### Title: Class "nMatrix" of Non-zero Pattern Matrices
### Aliases: nMatrix-class Arith,logical,nMatrix-method
###   Arith,nMatrix,logical-method Arith,nMatrix,numeric-method
###   Arith,numeric,nMatrix-method Compare,logical,nMatrix-method
###   Compare,nMatrix,logical-method Compare,nMatrix,nMatrix-method
###   Compare,nMatrix,numeric-method Compare,numeric,nMatrix-method
###   Logic,logical,nMatrix-method Logic,nMatrix,Matrix-method
###   Logic,nMatrix,logical-method Logic,nMatrix,nMatrix-method
###   Logic,nMatrix,numeric-method Logic,nMatrix,sparseVector-method
###   Logic,numeric,nMatrix-method Ops,nMatrix,dMatrix-method
###   Ops,nMatrix,lMatrix-method Ops,nMatrix,nMatrix-method
###   Ops,nMatrix,numeric-method Ops,numeric,nMatrix-method
###   Summary,nMatrix-method coerce,matrix,nMatrix-method
###   coerce,numLike,nMatrix-method
### Keywords: classes algebra

### ** Examples

getClass("nMatrix")

L3 <- Matrix(upper.tri(diag(3)))
L3 # an "ltCMatrix"
as(L3, "nMatrix") # -> ntC*

## similar, not using Matrix()
as(upper.tri(diag(3)), "nMatrix")# currently "ngTMatrix"



cleanEx()
nameEx("ndenseMatrix-class")
### * ndenseMatrix-class

flush(stderr()); flush(stdout())

### Name: ndenseMatrix-class
### Title: Virtual Class "ndenseMatrix" of Dense Logical Matrices
### Aliases: ndenseMatrix-class &,ndenseMatrix,ddiMatrix-method
###   &,ndenseMatrix,ldiMatrix-method *,ndenseMatrix,ddiMatrix-method
###   *,ndenseMatrix,ldiMatrix-method Ops,ndenseMatrix,ndenseMatrix-method
###   Summary,ndenseMatrix-method ^,ndenseMatrix,ddiMatrix-method
###   ^,ndenseMatrix,ldiMatrix-method coerce,matrix,ndenseMatrix-method
###   coerce,numLike,ndenseMatrix-method which,ndenseMatrix-method
### Keywords: classes

### ** Examples

showClass("ndenseMatrix")

as(diag(3) > 0, "ndenseMatrix")# -> "nge"



cleanEx()
nameEx("nearPD")
### * nearPD

flush(stderr()); flush(stdout())

### Name: nearPD
### Title: Nearest Positive Definite Matrix
### Aliases: nearPD
### Keywords: algebra array

### ** Examples

 ## Higham(2002), p.334f - simple example
 A <- matrix(1, 3,3); A[1,3] <- A[3,1] <- 0
 n.A <- nearPD(A, corr=TRUE, do2eigen=FALSE)
 n.A[c("mat", "normF")]
 n.A.m <- nearPD(A, corr=TRUE, do2eigen=FALSE, base.matrix=TRUE)$mat
 stopifnot(exprs = {                           #=--------------
   all.equal(n.A$mat[1,2], 0.760689917)
   all.equal(n.A$normF, 0.52779033, tolerance=1e-9)
   all.equal(n.A.m, unname(as.matrix(n.A$mat)), tolerance = 1e-15)# seen rel.d.= 1.46e-16
 })
 set.seed(27)
 m <- matrix(round(rnorm(25),2), 5, 5)
 m <- m + t(m)
 diag(m) <- pmax(0, diag(m)) + 1
 (m <- round(cov2cor(m), 2))

 str(near.m <- nearPD(m, trace = TRUE))
 round(near.m$mat, 2)
 norm(m - near.m$mat) # 1.102 / 1.08

 if(require("sfsmisc")) {
    m2 <- posdefify(m) # a simpler approach
    norm(m - m2)  # 1.185, i.e., slightly "less near"
 }

 round(nearPD(m, only.values=TRUE), 9)

## A longer example, extended from Jens' original,
## showing the effects of some of the options:

pr <- Matrix(c(1,     0.477, 0.644, 0.478, 0.651, 0.826,
               0.477, 1,     0.516, 0.233, 0.682, 0.75,
               0.644, 0.516, 1,     0.599, 0.581, 0.742,
               0.478, 0.233, 0.599, 1,     0.741, 0.8,
               0.651, 0.682, 0.581, 0.741, 1,     0.798,
               0.826, 0.75,  0.742, 0.8,   0.798, 1),
             nrow = 6, ncol = 6)

nc.  <- nearPD(pr, conv.tol = 1e-7) # default
nc.$iterations  # 2
nc.1 <- nearPD(pr, conv.tol = 1e-7, corr = TRUE)
nc.1$iterations # 11 / 12 (!)
ncr   <- nearPD(pr, conv.tol = 1e-15)
str(ncr)# still 2 iterations
ncr.1 <- nearPD(pr, conv.tol = 1e-15, corr = TRUE)
ncr.1 $ iterations # 27 / 30 !

ncF <- nearPD(pr, conv.tol = 1e-15, conv.norm = "F")
stopifnot(all.equal(ncr, ncF))# norm type does not matter at all in this example

## But indeed, the 'corr = TRUE' constraint did ensure a better solution;
## cov2cor() does not just fix it up equivalently :
norm(pr - cov2cor(ncr$mat)) # = 0.09994
norm(pr -       ncr.1$mat)  # = 0.08746 / 0.08805

### 3) a real data example from a 'systemfit' model (3 eq.):
(load(system.file("external", "symW.rda", package="Matrix"))) # "symW"
dim(symW) #  24 x 24
class(symW)# "dsCMatrix": sparse symmetric
if(dev.interactive())  image(symW)
EV <- eigen(symW, only=TRUE)$values
summary(EV) ## looking more closely {EV sorted decreasingly}:
tail(EV)# all 6 are negative
EV2 <- eigen(sWpos <- nearPD(symW)$mat, only=TRUE)$values
stopifnot(EV2 > 0)
if(require("sfsmisc")) {
       plot(pmax(1e-3,EV), EV2, type="o", log="xy", xaxt="n",yaxt="n")
       eaxis(1); eaxis(2)
} else plot(pmax(1e-3,EV), EV2, type="o", log="xy")
abline(0,1, col="red3",lty=2)




cleanEx()
nameEx("ngeMatrix-class")
### * ngeMatrix-class

flush(stderr()); flush(stdout())

### Name: ngeMatrix-class
### Title: Class "ngeMatrix" of General Dense Nonzero-pattern Matrices
### Aliases: ngeMatrix-class !,ngeMatrix-method
###   Arith,ngeMatrix,ngeMatrix-method Compare,ngeMatrix,ngeMatrix-method
###   Logic,ngeMatrix,ngeMatrix-method as.vector,ngeMatrix-method
###   coerce,ngeMatrix,matrix-method coerce,ngeMatrix,vector-method
### Keywords: classes

### ** Examples

showClass("ngeMatrix")
## "lgeMatrix" is really more relevant



cleanEx()
nameEx("nnzero")
### * nnzero

flush(stderr()); flush(stdout())

### Name: nnzero
### Title: The Number of Non-Zero Values of a Matrix
### Aliases: nnzero nnzero,ANY-method nnzero,CHMfactor-method
###   nnzero,array-method nnzero,denseMatrix-method
###   nnzero,diagonalMatrix-method nnzero,indMatrix-method
###   nnzero,sparseMatrix-method nnzero,vector-method
### Keywords: attribute

### ** Examples

m <- Matrix(0+1:28, nrow = 4)
m[-3,c(2,4:5,7)] <- m[ 3, 1:4] <- m[1:3, 6] <- 0
(mT <- as(m, "TsparseMatrix"))
nnzero(mT)
(S <- crossprod(mT))
nnzero(S)
str(S) # slots are smaller than nnzero()
stopifnot(nnzero(S) == sum(as.matrix(S) != 0))# failed earlier

data(KNex)
M <- KNex$mm
class(M)
dim(M)
length(M); stopifnot(length(M) == prod(dim(M)))
nnzero(M) # more relevant than length
## the above are also visible from
str(M)



cleanEx()
nameEx("norm")
### * norm

flush(stderr()); flush(stdout())

### Name: norm
### Title: Matrix Norms
### Aliases: norm norm,ANY,missing-method norm,denseMatrix,character-method
###   norm,dgeMatrix,character-method norm,diagonalMatrix,character-method
###   norm,dspMatrix,character-method norm,dsyMatrix,character-method
###   norm,dtpMatrix,character-method norm,dtrMatrix,character-method
###   norm,sparseMatrix,character-method
### Keywords: algebra

### ** Examples

x <- Hilbert(9)
norm(x)# = "O" = "1"
stopifnot(identical(norm(x), norm(x, "1")))
norm(x, "I")# the same, because 'x' is symmetric

allnorms <- function(d) vapply(c("1","I","F","M","2"),
                               norm, x = d, double(1))
allnorms(x)
allnorms(Hilbert(10))

i <- c(1,3:8); j <- c(2,9,6:10); x <- 7 * (1:7)
A <- sparseMatrix(i, j, x = x)                      ##  8 x 10 "dgCMatrix"
(sA <- sparseMatrix(i, j, x = x, symmetric = TRUE)) ## 10 x 10 "dsCMatrix"
(tA <- sparseMatrix(i, j, x = x, triangular= TRUE)) ## 10 x 10 "dtCMatrix"
(allnorms(A) -> nA)
allnorms(sA)
allnorms(tA)
stopifnot(all.equal(nA, allnorms(as(A, "matrix"))),
	  all.equal(nA, allnorms(tA))) # because tA == rbind(A, 0, 0)
A. <- A; A.[1,3] <- NA
stopifnot(is.na(allnorms(A.))) # gave error



cleanEx()
nameEx("nsparseMatrix-classes")
### * nsparseMatrix-classes

flush(stderr()); flush(stdout())

### Name: nsparseMatrix-classes
### Title: Sparse "pattern" Matrices
### Aliases: nsparseMatrix-class ngCMatrix-class ngRMatrix-class
###   ngTMatrix-class ntCMatrix-class ntRMatrix-class ntTMatrix-class
###   nsCMatrix-class nsRMatrix-class nsTMatrix-class
###   !,nsparseMatrix-method -,nsparseMatrix,missing-method
###   Arith,nsparseMatrix,Matrix-method
###   Arith,dsparseMatrix,nsparseMatrix-method
###   Arith,lsparseMatrix,nsparseMatrix-method
###   Arith,nsparseMatrix,dsparseMatrix-method
###   Arith,nsparseMatrix,lsparseMatrix-method
###   Ops,nsparseMatrix,dsparseMatrix-method
###   Ops,nsparseMatrix,lsparseMatrix-method
###   Ops,nsparseMatrix,sparseMatrix-method as.logical,nsparseMatrix-method
###   as.numeric,nsparseMatrix-method coerce,matrix,nsparseMatrix-method
###   coerce,nsparseMatrix,dMatrix-method
###   coerce,nsparseMatrix,dsparseMatrix-method
###   coerce,nsparseMatrix,indMatrix-method
###   coerce,nsparseMatrix,lMatrix-method
###   coerce,nsparseMatrix,lsparseMatrix-method
###   coerce,nsparseMatrix,pMatrix-method
###   coerce,numLike,nsparseMatrix-method which,nsparseMatrix-method
###   which,ngTMatrix-method which,ntTMatrix-method
###   coerce,nsCMatrix,RsparseMatrix-method
###   coerce,nsRMatrix,CsparseMatrix-method which,nsTMatrix-method
### Keywords: classes algebra

### ** Examples

(m <- Matrix(c(0,0,2:0), 3,5, dimnames=list(LETTERS[1:3],NULL)))
## ``extract the nonzero-pattern of (m) into an nMatrix'':
nm <- as(m, "nsparseMatrix") ## -> will be a "ngCMatrix"
str(nm) # no 'x' slot
nnm <- !nm # no longer sparse
## consistency check:
stopifnot(xor(as( nm, "matrix"),
              as(nnm, "matrix")))

## low-level way of adding "non-structural zeros" :
nnm <- as(nnm, "lsparseMatrix") # "lgCMatrix"
nnm@x[2:4] <- c(FALSE, NA, NA)
nnm
as(nnm, "nMatrix") # NAs *and* non-structural 0  |--->  'TRUE'

data(KNex)
nmm <- as(KNex $ mm, "nMatrix")
str(xlx <- crossprod(nmm))# "nsCMatrix"
stopifnot(isSymmetric(xlx))
image(xlx, main=paste("crossprod(nmm) : Sparse", class(xlx)))



cleanEx()
nameEx("nsyMatrix-class")
### * nsyMatrix-class

flush(stderr()); flush(stdout())

### Name: nsyMatrix-class
### Title: Symmetric Dense Nonzero-Pattern Matrices
### Aliases: nsyMatrix-class nspMatrix-class !,nsyMatrix-method
###   !,nspMatrix-method
### Keywords: classes

### ** Examples

(s0 <- new("nsyMatrix"))

(M2 <- Matrix(c(TRUE, NA, FALSE, FALSE), 2, 2)) # logical dense (ltr)
(sM <- M2 & t(M2)) # "lge"
class(sM <- as(sM, "nMatrix"))         # -> "nge"
     (sM <- as(sM, "symmetricMatrix")) # -> "nsy"
str  (sM <- as(sM, "packedMatrix"))    # -> "nsp": packed symmetric



cleanEx()
nameEx("ntrMatrix-class")
### * ntrMatrix-class

flush(stderr()); flush(stdout())

### Name: ntrMatrix-class
### Title: Triangular Dense Logical Matrices
### Aliases: ntrMatrix-class ntpMatrix-class !,ntrMatrix-method
###   !,ntpMatrix-method
### Keywords: classes

### ** Examples

showClass("ntrMatrix")

str(new("ntpMatrix"))
(nutr <- as(upper.tri(matrix(, 4, 4)), "ndenseMatrix"))
str(nutp <- pack(nutr)) # packed matrix: only 10 = 4*(4+1)/2 entries
!nutp # the logical negation (is *not* logical triangular !)
## but this one is:
stopifnot(all.equal(nutp, pack(!!nutp)))



cleanEx()
nameEx("number-class")
### * number-class

flush(stderr()); flush(stdout())

### Name: number-class
### Title: Class "number" of Possibly Complex Numbers
### Aliases: number-class
### Keywords: classes

### ** Examples

showClass("number")
stopifnot( is(1i, "number"), is(pi, "number"), is(1:3, "number") )



cleanEx()
nameEx("pMatrix-class")
### * pMatrix-class

flush(stderr()); flush(stdout())

### Name: pMatrix-class
### Title: Permutation matrices
### Aliases: pMatrix-class -,pMatrix,missing-method
###   coerce,integer,pMatrix-method coerce,matrix,pMatrix-method
###   coerce,numeric,pMatrix-method determinant,pMatrix,logical-method
###   t,pMatrix-method
### Keywords: classes

### ** Examples

(pm1 <- as(as.integer(c(2,3,1)), "pMatrix"))
t(pm1) # is the same as
solve(pm1)
pm1 %*% t(pm1) # check that the transpose is the inverse
stopifnot(all(diag(3) == as(pm1 %*% t(pm1), "matrix")),
          is.logical(as(pm1, "matrix")))

set.seed(11)
## random permutation matrix :
(p10 <- as(sample(10),"pMatrix"))

## Permute rows / columns of a numeric matrix :
(mm <- round(array(rnorm(3 * 3), c(3, 3)), 2))
mm %*% pm1
pm1 %*% mm
try(as(as.integer(c(3,3,1)), "pMatrix"))# Error: not a permutation

as(pm1, "TsparseMatrix")
p10[1:7, 1:4] # gives an "ngTMatrix" (most economic!)

## row-indexing of a <pMatrix> keeps it as an <indMatrix>:
p10[1:3, ]



cleanEx()
nameEx("packedMatrix-class")
### * packedMatrix-class

flush(stderr()); flush(stdout())

### Name: packedMatrix-class
### Title: Virtual Class '"packedMatrix"' of Packed Dense Matrices
### Aliases: packedMatrix-class coerce,matrix,packedMatrix-method
###   diag,packedMatrix-method diag<-,packedMatrix-method
###   t,packedMatrix-method
### Keywords: classes

### ** Examples

showClass("packedMatrix")
showMethods(classes = "packedMatrix")



cleanEx()
nameEx("printSpMatrix")
### * printSpMatrix

flush(stderr()); flush(stdout())

### Name: printSpMatrix
### Title: Format and Print Sparse Matrices Flexibly
### Aliases: formatSpMatrix printSpMatrix printSpMatrix2
### Keywords: print

### ** Examples

f1 <- gl(5, 3, labels = LETTERS[1:5])
X <- as(f1, "sparseMatrix")
X ## <==>  show(X)  <==>  print(X)
t(X) ## shows column names, since only 5 columns

X2 <- as(gl(12, 3, labels = paste(LETTERS[1:12],"c",sep=".")),
         "sparseMatrix")
X2
## less nice, but possible:
print(X2, col.names = TRUE) # use [,1] [,2] .. => does not fit

## Possibilities with column names printing:
      t(X2) # suppressing column names
print(t(X2), col.names=TRUE)
print(t(X2), zero.print = "", col.names="abbr. 1")
print(t(X2), zero.print = "-", col.names="substring 2")

## Don't show: 
op <- options(max.print = 25000, width = 80)
sink(print(tempfile()))
M <- Matrix(0, 10000, 100)
M[1,1] <- M[2,3] <- 3.14
st <- system.time(show(M))
sink()
st

stopifnot(st[1] < 1.0) # only 0.09 on cmath-3
options(op)
## End(Don't show)



cleanEx()
nameEx("qr-methods")
### * qr-methods

flush(stderr()); flush(stdout())

### Name: qr-methods
### Title: QR Decomposition - S4 Methods and Generic
### Aliases: qr qr-methods qrR qr,denseMatrix-method qr,dgCMatrix-method
###   qr,sparseMatrix-method
### Keywords: methods algebra array

### ** Examples

##------------- example of pivoting -- from base'  qraux.Rd -------------
X <- cbind(int = 1,
           b1=rep(1:0, each=3), b2=rep(0:1, each=3),
           c1=rep(c(1,0,0), 2), c2=rep(c(0,1,0), 2), c3=rep(c(0,0,1),2))
rownames(X) <- paste0("r", seq_len(nrow(X)))
dnX <- dimnames(X)
bX <- X # [b]ase version of X
X <- as(bX, "sparseMatrix")
X # is singular, columns "b2" and "c3" are "extra"
stopifnot(identical(dimnames(X), dnX))# some versions changed X's dimnames!
c(rankMatrix(X)) # = 4 (not 6)
m <- function(.) as(., "matrix")

##----- regular case ------------------------------------------
Xr <- X[ , -c(3,6)] # the "regular" (non-singular) version of X
stopifnot(rankMatrix(Xr) == ncol(Xr))
Y <- cbind(y <- setNames(1:6, paste0("y", 1:6)))
## regular case:
qXr   <- qr(  Xr)
qxr   <- qr(m(Xr))
qxrLA <- qr(m(Xr), LAPACK=TRUE) # => qr.fitted(), qr.resid() not supported
qcfXy <- qr.coef (qXr, y) # vector
qcfXY <- qr.coef (qXr, Y) # 4x1 dgeMatrix
cf <- c(int=6, b1=-3, c1=-2, c2=-1)
doExtras <- interactive() || nzchar(Sys.getenv("R_MATRIX_CHECK_EXTRA"))
tolE <- if(doExtras) 1e-15 else 1e-13
stopifnot(exprs = {
  all.equal(qr.coef(qxr,  y),   cf,  tol=tolE)
  all.equal(qr.coef(qxrLA,y),   cf,  tol=tolE)
  all.equal(qr.coef(qxr,  Y), m(cf), tol=tolE)
  all.equal(  qcfXy,    cf, tol=tolE)
  all.equal(m(qcfXY), m(cf), tol=tolE)
  all.equal(y, qr.fitted(qxr, y), tol=2*tolE)
  all.equal(y, qr.fitted(qXr, y), tol=2*tolE)
  all.equal(m(qr.fitted(qXr, Y)), qr.fitted(qxr, Y), tol=tolE)
  all.equal(  qr.resid (qXr, y),  qr.resid (qxr, y), tol=tolE)
  all.equal(m(qr.resid (qXr, Y)), qr.resid (qxr, Y), tol=tolE)
})

##----- rank-deficient ("singular") case ------------------------------------

(qX <- qr(  X))           # both @p and @q are non-trivial permutations
 qx <- qr(m(X)) ; str(qx) # $pivot is non-trivial, too

drop0(R. <- qr.R(qX), tol=tolE) # columns *permuted*: c3 b1 ..
Q. <- qr.Q(qX)
qI <- sort.list(qX@q) # the inverse 'q' permutation
(X. <- drop0(Q. %*% R.[, qI], tol=tolE))## just = X, incl. correct colnames
stopifnot(all(X - X.) < 8*.Machine$double.eps,
          ## qrR(.) returns R already "back permuted" (as with qI):
          identical(R.[, qI], qrR(qX)) )
##
## In this sense, classical qr.coef() is fine:
cfqx <- qr.coef(qx, y) # quite different from
nna <- !is.na(cfqx)
stopifnot(all.equal(unname(qr.fitted(qx,y)),
                    as.numeric(X[,nna] %*% cfqx[nna])))
## FIXME: do these make *any* sense? --- should give warnings !
qr.coef(qX, y)
qr.coef(qX, Y)
rm(m)



cleanEx()
nameEx("rankMatrix")
### * rankMatrix

flush(stderr()); flush(stdout())

### Name: rankMatrix
### Title: Rank of a Matrix
### Aliases: rankMatrix qr2rankMatrix
### Keywords: algebra array

### ** Examples

rankMatrix(cbind(1, 0, 1:3)) # 2

(meths <- eval(formals(rankMatrix)$method))

## a "border" case:
H12 <- Hilbert(12)
rankMatrix(H12, tol = 1e-20) # 12;  but  11  with default method & tol.
sapply(meths, function(.m.) rankMatrix(H12, method = .m.))
## tolNorm2   qr.R  qrLINPACK   qr  useGrad maybeGrad
##       11     11         12   12       11        11
## The meaning of 'tol' for method="qrLINPACK" and *dense* x is not entirely "scale free"
rMQL <- function(ex, M) rankMatrix(M, method="qrLINPACK",tol = 10^-ex)
rMQR <- function(ex, M) rankMatrix(M, method="qr.R",     tol = 10^-ex)
sapply(5:15, rMQL, M = H12) # result is platform dependent
##  7  7  8 10 10 11 11 11 12 12 12  {x86_64}
sapply(5:15, rMQL, M = 1000 * H12) # not identical unfortunately
##  7  7  8 10 11 11 12 12 12 12 12
sapply(5:15, rMQR, M = H12)
##  5  6  7  8  8  9  9 10 10 11 11
sapply(5:15, rMQR, M = 1000 * H12) # the *same*
## Don't show: 
  (r12 <- sapply(5:15, rMQR, M = H12))
  stopifnot(identical(r12, sapply(5:15, rMQR, M = H12 / 100)),
            identical(r12, sapply(5:15, rMQR, M = H12 * 1e5)))

  rM1 <- function(ex, M) rankMatrix(M, tol = 10^-ex)
  (r12 <- sapply(5:15, rM1, M = H12))
  stopifnot(identical(r12, sapply(5:15, rM1, M = H12 / 100)),
            identical(r12, sapply(5:15, rM1, M = H12 * 1e5)))
## End(Don't show)

## "sparse" case:
M15 <- kronecker(diag(x=c(100,1,10)), Hilbert(5))
sapply(meths, function(.m.) rankMatrix(M15, method = .m.))
#--> all 15, but 'useGrad' has 14.
sapply(meths, function(.m.) rankMatrix(M15, method = .m., tol = 1e-7)) # all 14

## "large" sparse
n <- 250000; p <- 33; nnz <- 10000
L <- sparseMatrix(i = sample.int(n, nnz, replace=TRUE),
                  j = sample.int(p, nnz, replace=TRUE), x = rnorm(nnz))
(st1 <- system.time(r1 <- rankMatrix(L)))                # warning+ ~1.5 sec (2013)
(st2 <- system.time(r2 <- rankMatrix(L, method = "qr"))) # considerably faster!
r1[[1]] == print(r2[[1]]) ## -->  ( 33  TRUE )
## Don't show: 
stopifnot(r1[[1]] == 33, 33 == r2[[1]])
if(interactive() || nzchar(Sys.getenv("R_MATRIX_CHECK_EXTRA")))
    stopifnot(st2[[1]] < 0.2) # seeing 0.03 (on ~ 2010-hardware; R 3.0.2)
## End(Don't show)
## another sparse-"qr" one, which ``failed'' till 2013-11-23:
set.seed(42)
f1 <- factor(sample(50, 1000, replace=TRUE))
f2 <- factor(sample(50, 1000, replace=TRUE))
f3 <- factor(sample(50, 1000, replace=TRUE))
D <- t(do.call(rbind, lapply(list(f1,f2,f3), as, 'sparseMatrix')))
dim(D); nnzero(D) ## 1000 x 150 // 3000 non-zeros (= 2%)
stopifnot(rankMatrix(D,           method='qr') == 148,
	  rankMatrix(crossprod(D),method='qr') == 148)

## zero matrix has rank 0 :
stopifnot(sapply(meths, function(.m.)
                        rankMatrix(matrix(0, 2, 2), method = .m.)) == 0)



cleanEx()
nameEx("rcond")
### * rcond

flush(stderr()); flush(stdout())

### Name: rcond
### Title: Estimate the Reciprocal Condition Number
### Aliases: rcond rcond,ANY,missing-method
###   rcond,denseMatrix,character-method rcond,dgeMatrix,character-method
###   rcond,dpoMatrix,character-method rcond,dppMatrix,character-method
###   rcond,dspMatrix,character-method rcond,dsyMatrix,character-method
###   rcond,dtpMatrix,character-method rcond,dtrMatrix,character-method
###   rcond,sparseMatrix,character-method
### Keywords: array algebra

### ** Examples

x <- Matrix(rnorm(9), 3, 3)
rcond(x)
## typically "the same" (with more computational effort):
1 / (norm(x) * norm(solve(x)))
rcond(Hilbert(9))  # should be about 9.1e-13

## For non-square matrices:
rcond(x1 <- cbind(1,1:10))# 0.05278
rcond(x2 <- cbind(x1, 2:11))# practically 0, since x2 does not have full rank

## sparse
(S1 <- Matrix(rbind(0:1,0, diag(3:-2))))
rcond(S1)
m1 <- as(S1, "denseMatrix")
all.equal(rcond(S1), rcond(m1))

## wide and sparse
rcond(Matrix(cbind(0, diag(2:-1))))

## Large sparse example ----------
m <- Matrix(c(3,0:2), 2,2)
M <- bdiag(kronecker(Diagonal(2), m), kronecker(m,m))
36*(iM <- solve(M)) # still sparse
MM <- kronecker(Diagonal(10), kronecker(Diagonal(5),kronecker(m,M)))
dim(M3 <- kronecker(bdiag(M,M),MM)) # 12'800 ^ 2
if(interactive()) ## takes about 2 seconds if you have >= 8 GB RAM
  system.time(r <- rcond(M3))
## whereas this is *fast* even though it computes  solve(M3)
system.time(r. <- rcond(M3, useInv=TRUE))
if(interactive()) ## the values are not the same
  c(r, r.)  # 0.05555 0.013888
## for all 4 norms available for sparseMatrix :
cbind(rr <- sapply(c("1","I","F","M"),
             function(N) rcond(M3, norm=N, useInv=TRUE)))
## Don't show: 
stopifnot(all.equal(r., 1/72, tolerance=1e-12))
## End(Don't show)



cleanEx()
nameEx("rep2abI")
### * rep2abI

flush(stderr()); flush(stdout())

### Name: rep2abI
### Title: Replicate Vectors into 'abIndex' Result
### Aliases: rep2abI
### Keywords: manip

### ** Examples

(ab <- rep2abI(2:7, 4))
stopifnot(identical(as(ab, "numeric"),
	   rep(2:7, 4)))



cleanEx()
nameEx("replValue-class")
### * replValue-class

flush(stderr()); flush(stdout())

### Name: replValue-class
### Title: Virtual Class "replValue" - Simple Class for Subassignment
###   Values
### Aliases: replValue-class
### Keywords: classes

### ** Examples

showClass("replValue")



cleanEx()
nameEx("rleDiff-class")
### * rleDiff-class

flush(stderr()); flush(stdout())

### Name: rleDiff-class
### Title: Class "rleDiff" of rle(diff(.)) Stored Vectors
### Aliases: rleDiff-class show,rleDiff-method
### Keywords: classes

### ** Examples

showClass("rleDiff")

ab <- c(abIseq(2, 100), abIseq(20, -2))
ab@rleD  # is "rleDiff"



cleanEx()
nameEx("rsparsematrix")
### * rsparsematrix

flush(stderr()); flush(stdout())

### Name: rsparsematrix
### Title: Random Sparse Matrix
### Aliases: rsparsematrix
### Keywords: array distribution

### ** Examples

set.seed(17)# to be reproducible
M <- rsparsematrix(8, 12, nnz = 30) # small example, not very sparse
M
M1 <- rsparsematrix(1000, 20,  nnz = 123,  rand.x = runif)
summary(M1)

## a random *symmetric* Matrix
(S9 <- rsparsematrix(9, 9, nnz = 10, symmetric=TRUE)) # dsCMatrix
nnzero(S9)# ~ 20: as 'nnz' only counts one "triangle"

## a random patter*n* aka boolean Matrix (no 'x' slot):
(n7 <- rsparsematrix(5, 12, nnz = 10, rand.x = NULL))

## a [T]riplet representation sparseMatrix:
T2 <- rsparsematrix(40, 12, nnz = 99, repr = "T")
head(T2)



cleanEx()
nameEx("solve-methods")
### * solve-methods

flush(stderr()); flush(stdout())

### Name: solve-methods
### Title: Methods in Package Matrix for Function 'solve()'
### Aliases: solve solve-methods solve,ANY,ANY-method
###   solve,CHMfactor,denseMatrix-method
###   solve,CHMfactor,sparseMatrix-method solve,CHMfactor,matrix-method
###   solve,CHMfactor,missing-method solve,CHMfactor,numLike-method
###   solve,CsparseMatrix,ANY-method solve,Matrix,sparseVector-method
###   solve,MatrixFactorization,ANY-method
###   solve,MatrixFactorization,missing-method
###   solve,MatrixFactorization,sparseVector-method
###   solve,RsparseMatrix,ANY-method solve,TsparseMatrix,ANY-method
###   solve,ddiMatrix,Matrix-method solve,ddiMatrix,matrix-method
###   solve,ddiMatrix,missing-method solve,ddiMatrix,numLike-method
###   solve,denseLU,missing-method solve,denseMatrix,ANY-method
###   solve,dgCMatrix,denseMatrix-method solve,dgCMatrix,matrix-method
###   solve,dgCMatrix,missing-method solve,dgCMatrix,numLike-method
###   solve,dgCMatrix,sparseMatrix-method solve,dgeMatrix,Matrix-method
###   solve,dgeMatrix,matrix-method solve,dgeMatrix,missing-method
###   solve,dgeMatrix,numLike-method solve,diagonalMatrix,ANY-method
###   solve,dpoMatrix,Matrix-method solve,dpoMatrix,matrix-method
###   solve,dpoMatrix,missing-method solve,dpoMatrix,numLike-method
###   solve,dppMatrix,Matrix-method solve,dppMatrix,matrix-method
###   solve,dppMatrix,missing-method solve,dppMatrix,numLike-method
###   solve,dsCMatrix,denseMatrix-method solve,dsCMatrix,matrix-method
###   solve,dsCMatrix,missing-method solve,dsCMatrix,numLike-method
###   solve,dsCMatrix,sparseMatrix-method solve,dspMatrix,Matrix-method
###   solve,dspMatrix,matrix-method solve,dspMatrix,missing-method
###   solve,dspMatrix,numLike-method solve,dsyMatrix,Matrix-method
###   solve,dsyMatrix,matrix-method solve,dsyMatrix,missing-method
###   solve,dsyMatrix,numLike-method solve,dtpMatrix,Matrix-method
###   solve,dtpMatrix,matrix-method solve,dtpMatrix,missing-method
###   solve,dtpMatrix,numLike-method solve,dtrMatrix,Matrix-method
###   solve,dtrMatrix,matrix-method solve,dtrMatrix,missing-method
###   solve,dtrMatrix,numLike-method solve,indMatrix,ANY-method
###   solve,pMatrix,Matrix-method solve,pMatrix,matrix-method
###   solve,pMatrix,missing-method solve,pMatrix,numLike-method
###   solve,dtCMatrix,dgCMatrix-method solve,dtCMatrix,dgeMatrix-method
###   solve,dtCMatrix,dsCMatrix-method solve,dtCMatrix,dspMatrix-method
###   solve,dtCMatrix,dsyMatrix-method solve,dtCMatrix,dtCMatrix-method
###   solve,dtCMatrix,dtpMatrix-method solve,dtCMatrix,dtrMatrix-method
###   solve,dtCMatrix,denseMatrix-method solve,dtCMatrix,matrix-method
###   solve,dtCMatrix,missing-method solve,dtCMatrix,numLike-method
###   solve,dtCMatrix,sparseMatrix-method solve,matrix,Matrix-method
###   solve,matrix,sparseVector-method solve,sparseQR,ANY-method
###   solve,sparseQR,missing-method
### Keywords: methods

### ** Examples

## A close to symmetric example with "quite sparse" inverse:
n1 <- 7; n2 <- 3
dd <- data.frame(a = gl(n1,n2), b = gl(n2,1,n1*n2))# balanced 2-way
X <- sparse.model.matrix(~ -1+ a + b, dd)# no intercept --> even sparser
XXt <- tcrossprod(X)
diag(XXt) <- rep(c(0,0,1,0), length.out = nrow(XXt))

n <- nrow(ZZ <- kronecker(XXt, Diagonal(x=c(4,1))))
image(a <- 2*Diagonal(n) + ZZ %*% Diagonal(x=c(10, rep(1, n-1))))
isSymmetric(a) # FALSE
image(drop0(skewpart(a)))
image(ia0 <- solve(a)) # checker board, dense [but really, a is singular!]
try(solve(a, sparse=TRUE))##-> error [ TODO: assertError ]
ia. <- solve(a, sparse=TRUE, tol = 1e-19)##-> *no* error
if(R.version$arch == "x86_64")
  ## Fails on 32-bit [Fedora 19, R 3.0.2] from Matrix 1.1-0 on [FIXME ??] only
  stopifnot(all.equal(as.matrix(ia.), as.matrix(ia0)))
a <- a + Diagonal(n)
iad <- solve(a)
ias <- solve(a, sparse=TRUE)
stopifnot(all.equal(as(ias,"denseMatrix"), iad, tolerance=1e-14))
I. <- iad %*% a          ; image(I.)
I0 <- drop0(zapsmall(I.)); image(I0)
.I <- a %*% iad
.I0 <- drop0(zapsmall(.I))
stopifnot( all.equal(as(I0, "diagonalMatrix"), Diagonal(n)),
           all.equal(as(.I0,"diagonalMatrix"), Diagonal(n)) )




cleanEx()
nameEx("spMatrix")
### * spMatrix

flush(stderr()); flush(stdout())

### Name: spMatrix
### Title: Sparse Matrix Constructor From Triplet
### Aliases: spMatrix
### Keywords: array

### ** Examples

## simple example
A <- spMatrix(10,20, i = c(1,3:8),
                     j = c(2,9,6:10),
                     x = 7 * (1:7))
A # a "dgTMatrix"
summary(A)
str(A) # note that *internally* 0-based indices (i,j) are used

L <- spMatrix(9, 30, i = rep(1:9, 3), 1:27,
              (1:27) %% 4 != 1)
L # an "lgTMatrix"


## A simplified predecessor of  Matrix'  rsparsematrix() function :

 rSpMatrix <- function(nrow, ncol, nnz,
                       rand.x = function(n) round(rnorm(nnz), 2))
 {
     ## Purpose: random sparse matrix
     ## --------------------------------------------------------------
     ## Arguments: (nrow,ncol): dimension
     ##          nnz  :  number of non-zero entries
     ##         rand.x:  random number generator for 'x' slot
     ## --------------------------------------------------------------
     ## Author: Martin Maechler, Date: 14.-16. May 2007
     stopifnot((nnz <- as.integer(nnz)) >= 0,
               nrow >= 0, ncol >= 0, nnz <= nrow * ncol)
     spMatrix(nrow, ncol,
              i = sample(nrow, nnz, replace = TRUE),
              j = sample(ncol, nnz, replace = TRUE),
              x = rand.x(nnz))
 }

 M1 <- rSpMatrix(100000, 20, nnz = 200)
 summary(M1)



cleanEx()
nameEx("sparse.model.matrix")
### * sparse.model.matrix

flush(stderr()); flush(stdout())

### Name: sparse.model.matrix
### Title: Construct Sparse Design / Model Matrices
### Aliases: sparse.model.matrix fac2sparse fac2Sparse
### Keywords: models

### ** Examples

dd <- data.frame(a = gl(3,4), b = gl(4,1,12))# balanced 2-way
options("contrasts") # the default:  "contr.treatment"
sparse.model.matrix(~ a + b, dd)
sparse.model.matrix(~ -1+ a + b, dd)# no intercept --> even sparser
sparse.model.matrix(~ a + b, dd, contrasts = list(a="contr.sum"))
sparse.model.matrix(~ a + b, dd, contrasts = list(b="contr.SAS"))

## Sparse method is equivalent to the traditional one :
stopifnot(all(sparse.model.matrix(~ a + b, dd) ==
	      Matrix(model.matrix(~ a + b, dd), sparse=TRUE)),
	  all(sparse.model.matrix(~ 0+ a + b, dd) ==
	      Matrix(model.matrix(~ 0+ a + b, dd), sparse=TRUE)))

(ff <- gl(3,4,, c("X","Y", "Z")))
fac2sparse(ff) #  3 x 12 sparse Matrix of class "dgCMatrix"
##
##  X  1 1 1 1 . . . . . . . .
##  Y  . . . . 1 1 1 1 . . . .
##  Z  . . . . . . . . 1 1 1 1

## can also be computed via sparse.model.matrix():
f30 <- gl(3,0    )
f12 <- gl(3,0, 12)
stopifnot(
  all.equal(t( fac2sparse(ff) ),
	    sparse.model.matrix(~ 0+ff),
	    tolerance = 0, check.attributes=FALSE),
  is(M <- fac2sparse(f30, drop= TRUE),"CsparseMatrix"), dim(M) == c(0, 0),
  is(M <- fac2sparse(f30, drop=FALSE),"CsparseMatrix"), dim(M) == c(3, 0),
  is(M <- fac2sparse(f12, drop= TRUE),"CsparseMatrix"), dim(M) == c(0,12),
  is(M <- fac2sparse(f12, drop=FALSE),"CsparseMatrix"), dim(M) == c(3,12)
 )



cleanEx()
nameEx("sparseLU-class")
### * sparseLU-class

flush(stderr()); flush(stdout())

### Name: sparseLU-class
### Title: Sparse LU decomposition of a square sparse matrix
### Aliases: sparseLU-class
### Keywords: classes

### ** Examples

## Extending the one in   examples(lu), calling the matrix  A,
## and confirming the factorization identities :
A <- as(readMM(system.file("external/pores_1.mtx",
                            package = "Matrix")),
         "CsparseMatrix")
## with dimnames(.) - to see that they propagate to L, U :
dimnames(A) <- list(paste0("r", seq_len(nrow(A))),
                    paste0("C", seq_len(ncol(A))))
str(luA <- lu(A)) # p is a 0-based permutation of the rows
                  # q is a 0-based permutation of the columns
xA <- expand(luA)
## which is simply doing
stopifnot(identical(xA$ L, luA@L),
          identical(xA$ U, luA@U),
          identical(xA$ P, as(luA@p +1L, "pMatrix")),
          identical(xA$ Q, as(luA@q +1L, "pMatrix")))

P.LUQ <- with(xA, t(P) %*% L %*% U %*% Q)
stopifnot(all.equal(unname(A), unname(P.LUQ), tolerance = 1e-12))

## permute rows and columns of original matrix
pA <- A[luA@p + 1L, luA@q + 1L]
PAQ. <- with(xA, P %*% A %*% t(Q))
stopifnot(all.equal(unname(pA), unname(PAQ.), tolerance = 1e-12))

pLU <- drop0(luA@L %*% luA@U) # L %*% U -- dropping extra zeros
stopifnot(all.equal(pA, pLU, tolerance = 1e-12)) # (incl. permuted row- and column-names)



cleanEx()
nameEx("sparseMatrix-class")
### * sparseMatrix-class

flush(stderr()); flush(stdout())

### Name: sparseMatrix-class
### Title: Virtual Class "sparseMatrix" - Mother of Sparse Matrices
### Aliases: sparseMatrix-class -,sparseMatrix,missing-method
###   Math,sparseMatrix-method Ops,numeric,sparseMatrix-method
###   Ops,sparseMatrix,ddiMatrix-method Ops,sparseMatrix,ldiMatrix-method
###   Ops,sparseMatrix,nsparseMatrix-method Ops,sparseMatrix,numeric-method
###   Ops,sparseMatrix,sparseMatrix-method
###   cbind2,matrix,sparseMatrix-method
###   cbind2,sparseMatrix,diagonalMatrix-method
###   cbind2,sparseMatrix,matrix-method
###   cbind2,sparseMatrix,sparseMatrix-method
###   coerce,ANY,sparseMatrix-method coerce,factor,sparseMatrix-method
###   coerce,matrix,sparseMatrix-method coerce,numLike,sparseMatrix-method
###   coerce,sparseMatrix,sparseVector-method
###   coerce,table,sparseMatrix-method cov2cor,sparseMatrix-method
###   dim<-,sparseMatrix-method format,sparseMatrix-method
###   log,sparseMatrix-method mean,sparseMatrix-method
###   print,sparseMatrix-method rbind2,matrix,sparseMatrix-method
###   rbind2,sparseMatrix,diagonalMatrix-method
###   rbind2,sparseMatrix,matrix-method
###   rbind2,sparseMatrix,sparseMatrix-method rep,sparseMatrix-method
###   show,sparseMatrix-method summary,sparseMatrix-method
###   print.sparseMatrix
### Keywords: classes

### ** Examples

showClass("sparseMatrix") ## and look at the help() of its subclasses
M <- Matrix(0, 10000, 100)
M[1,1] <- M[2,3] <- 3.14
M  ## show(.) method suppresses printing of the majority of rows

data(CAex); dim(CAex) # 72 x 72 matrix
determinant(CAex) # works via sparse lu(.)

## factor -> t( <sparse design matrix> ) :
(fact <- gl(5, 3, 30, labels = LETTERS[1:5]))
(Xt <- as(fact, "sparseMatrix"))  # indicator rows

## missing values --> all-0 columns:
f.mis <- fact
i.mis <- c(3:5, 17)
is.na(f.mis) <- i.mis
Xt != (X. <- as(f.mis, "sparseMatrix")) # differ only in columns 3:5,17
stopifnot(all(X.[,i.mis] == 0), all(Xt[,-i.mis] == X.[,-i.mis]))



cleanEx()
nameEx("sparseMatrix")
### * sparseMatrix

flush(stderr()); flush(stdout())

### Name: sparseMatrix
### Title: General Sparse Matrix Construction from Nonzero Entries
### Aliases: sparseMatrix
### Keywords: array

### ** Examples

## simple example
i <- c(1,3:8); j <- c(2,9,6:10); x <- 7 * (1:7)
(A <- sparseMatrix(i, j, x = x))                    ##  8 x 10 "dgCMatrix"
summary(A)
str(A) # note that *internally* 0-based row indices are used

(sA <- sparseMatrix(i, j, x = x, symmetric = TRUE)) ## 10 x 10 "dsCMatrix"
(tA <- sparseMatrix(i, j, x = x, triangular= TRUE)) ## 10 x 10 "dtCMatrix"
stopifnot( all(sA == tA + t(tA)) ,
           identical(sA, as(tA + t(tA), "symmetricMatrix")))

## dims can be larger than the maximum row or column indices
(AA <- sparseMatrix(c(1,3:8), c(2,9,6:10), x = 7 * (1:7), dims = c(10,20)))
summary(AA)

## i, j and x can be in an arbitrary order, as long as they are consistent
set.seed(1); (perm <- sample(1:7))
(A1 <- sparseMatrix(i[perm], j[perm], x = x[perm]))
stopifnot(identical(A, A1))

## The slots are 0-index based, so
try( sparseMatrix(i=A@i, p=A@p, x= seq_along(A@x)) )
## fails and you should say so: 1-indexing is FALSE:
     sparseMatrix(i=A@i, p=A@p, x= seq_along(A@x), index1 = FALSE)

## the (i,j) pairs can be repeated, in which case the x's are summed
(args <- data.frame(i = c(i, 1), j = c(j, 2), x = c(x, 2)))
(Aa <- do.call(sparseMatrix, args))
## explicitly ask for elimination of such duplicates, so
## that the last one is used:
(A. <- do.call(sparseMatrix, c(args, list(use.last.ij = TRUE))))
stopifnot(Aa[1,2] == 9, # 2+7 == 9
          A.[1,2] == 2) # 2 was *after* 7

## for a pattern matrix, of course there is no "summing":
(nA <- do.call(sparseMatrix, args[c("i","j")]))

dn <- list(LETTERS[1:3], letters[1:5])
## pointer vectors can be used, and the (i,x) slots are sorted if necessary:
m <- sparseMatrix(i = c(3,1, 3:2, 2:1), p= c(0:2, 4,4,6), x = 1:6, dimnames = dn)
m
str(m)
stopifnot(identical(dimnames(m), dn))

sparseMatrix(x = 2.72, i=1:3, j=2:4) # recycling x
sparseMatrix(x = TRUE, i=1:3, j=2:4) # recycling x, |--> "lgCMatrix"

## no 'x' --> patter*n* matrix:
(n <- sparseMatrix(i=1:6, j=rev(2:7)))# -> ngCMatrix

## an empty sparse matrix:
(e <- sparseMatrix(dims = c(4,6), i={}, j={}))

## a symmetric one:
(sy <- sparseMatrix(i= c(2,4,3:5), j= c(4,7:5,5), x = 1:5,
                    dims = c(7,7), symmetric=TRUE))
stopifnot(isSymmetric(sy),
          identical(sy, ## switch i <-> j {and transpose }
    t( sparseMatrix(j= c(2,4,3:5), i= c(4,7:5,5), x = 1:5,
                    dims = c(7,7), symmetric=TRUE))))

## rsparsematrix() calls sparseMatrix() :
M1 <- rsparsematrix(1000, 20, nnz = 200)
summary(M1)

## pointers example in converting from other sparse matrix representations.
if(require(SparseM) && packageVersion("SparseM") >= 0.87 &&
   nzchar(dfil <- system.file("extdata", "rua_32_ax.rua", package = "SparseM"))) {
  X <- model.matrix(read.matrix.hb(dfil))
  XX <- sparseMatrix(j = X@ja, p = X@ia - 1L, x = X@ra, dims = X@dimension)
  validObject(XX)

  ## Alternatively, and even more user friendly :
  X. <- as(X, "Matrix")  # or also
  X2 <- as(X, "sparseMatrix")
  stopifnot(identical(XX, X.), identical(X., X2))
}



cleanEx()
nameEx("sparseQR-class")
### * sparseQR-class

flush(stderr()); flush(stdout())

### Name: sparseQR-class
### Title: Sparse QR decomposition of a sparse matrix
### Aliases: sparseQR-class qr.Q qr.Q,sparseQR-method qr.R,sparseQR-method
###   qr.coef,sparseQR,Matrix-method qr.coef,sparseQR,ddenseMatrix-method
###   qr.coef,sparseQR,matrix-method qr.coef,sparseQR,numeric-method
###   qr.fitted,sparseQR,Matrix-method
###   qr.fitted,sparseQR,ddenseMatrix-method
###   qr.fitted,sparseQR,matrix-method qr.fitted,sparseQR,numeric-method
###   qr.qty,sparseQR,Matrix-method qr.qty,sparseQR,ddenseMatrix-method
###   qr.qty,sparseQR,matrix-method qr.qty,sparseQR,numeric-method
###   qr.qy,sparseQR,Matrix-method qr.qy,sparseQR,ddenseMatrix-method
###   qr.qy,sparseQR,matrix-method qr.qy,sparseQR,numeric-method
###   qr.resid,sparseQR,Matrix-method qr.resid,sparseQR,ddenseMatrix-method
###   qr.resid,sparseQR,matrix-method qr.resid,sparseQR,numeric-method
### Keywords: classes algebra array

### ** Examples

data(KNex)
mm <- KNex $ mm
 y <- KNex $  y
 y. <- as(y, "CsparseMatrix")
str(qrm <- qr(mm))
 qc  <- qr.coef  (qrm, y); qc. <- qr.coef  (qrm, y.) # 2nd failed in Matrix <= 1.1-0
 qf  <- qr.fitted(qrm, y); qf. <- qr.fitted(qrm, y.)
 qs  <- qr.resid (qrm, y); qs. <- qr.resid (qrm, y.)
stopifnot(all.equal(qc, as.numeric(qc.),  tolerance=1e-12),
          all.equal(qf, as.numeric(qf.),  tolerance=1e-12),
          all.equal(qs, as.numeric(qs.),  tolerance=1e-12),
          all.equal(qf+qs, y, tolerance=1e-12))



cleanEx()
nameEx("sparseVector-class")
### * sparseVector-class

flush(stderr()); flush(stdout())

### Name: sparseVector-class
### Title: Sparse Vector Classes
### Aliases: sparseVector-class dsparseVector-class isparseVector-class
###   lsparseVector-class nsparseVector-class zsparseVector-class
###   xsparseVector-class c.sparseVector !,sparseVector-method
###   Arith,sparseVector,ddenseMatrix-method
###   Arith,sparseVector,dgeMatrix-method
###   Arith,sparseVector,sparseVector-method
###   Logic,sparseVector,dMatrix-method Logic,sparseVector,lMatrix-method
###   Logic,sparseVector,nMatrix-method
###   Logic,sparseVector,sparseVector-method Math,sparseVector-method
###   Math2,sparseVector-method Ops,ANY,sparseVector-method
###   Ops,sparseVector,ANY-method Ops,sparseVector,Matrix-method
###   Ops,sparseVector,atomicVector-method
###   Ops,sparseVector,sparseVector-method Summary,sparseVector-method
###   as.logical,sparseVector-method as.numeric,sparseVector-method
###   as.vector,sparseVector-method coerce,ANY,sparseVector-method
###   coerce,sparseVector,CsparseMatrix-method
###   coerce,sparseVector,Matrix-method
###   coerce,sparseVector,RsparseMatrix-method
###   coerce,sparseVector,TsparseMatrix-method
###   coerce,sparseVector,dsparseVector-method
###   coerce,sparseVector,integer-method
###   coerce,sparseVector,isparseVector-method
###   coerce,sparseVector,logical-method
###   coerce,sparseVector,lsparseVector-method
###   coerce,sparseVector,nsparseVector-method
###   coerce,sparseVector,numeric-method
###   coerce,sparseVector,sparseMatrix-method
###   coerce,sparseVector,vector-method
###   coerce,sparseVector,zsparseVector-method dim<-,sparseVector-method
###   head,sparseVector-method initialize,sparseVector-method
###   length,sparseVector-method log,sparseVector-method
###   mean,sparseVector-method rep,sparseVector-method
###   show,sparseVector-method t,sparseVector-method
###   tail,sparseVector-method toeplitz,sparseVector-method
###   -,dsparseVector,missing-method
###   Arith,dsparseVector,dsparseVector-method Math2,dsparseVector-method
###   Logic,lsparseVector,lsparseVector-method which,lsparseVector-method
###   Summary,nsparseVector-method coerce,ANY,nsparseVector-method
###   coerce,nsparseVector,dsparseVector-method
###   coerce,nsparseVector,isparseVector-method
###   coerce,nsparseVector,lsparseVector-method
###   coerce,nsparseVector,zsparseVector-method which,nsparseVector-method
### Keywords: classes

### ** Examples

getClass("sparseVector")
getClass("dsparseVector")
getClass("xsparseVector")# those with an 'x' slot

sx <- c(0,0,3, 3.2, 0,0,0,-3:1,0,0,2,0,0,5,0,0)
(ss <- as(sx, "sparseVector"))

ix <- as.integer(round(sx))
(is <- as(ix, "sparseVector")) ## an "isparseVector" (!)
(ns <- sparseVector(i= c(7, 3, 2), length = 10)) # "nsparseVector"
## rep() works too:
(ri <- rep(is, length.out= 25))

## Using `dim<-`  as in base R :
r <- ss
dim(r) <- c(4,5) # becomes a sparse Matrix:
r
## or coercion (as as.matrix() in base R):
as(ss, "Matrix")
stopifnot(all(ss == print(as(ss, "CsparseMatrix"))))

## currently has "non-structural" FALSE -- printing as ":"
(lis <- is & FALSE)
(nn <- is[is == 0]) # all "structural" FALSE

## NA-case
sN <- sx; sN[4] <- NA
(svN <- as(sN, "sparseVector"))

v <- as(c(0,0,3, 3.2, rep(0,9),-3,0,-1, rep(0,20),5,0),
         "sparseVector")
v <- rep(rep(v, 50), 5000)
set.seed(1); v[sample(v@i, 1e6)] <- 0
str(v)
system.time(for(i in 1:4) hv <- head(v, 1e6))
##   user  system elapsed
##  0.033   0.000   0.032
system.time(for(i in 1:4) h2 <- v[1:1e6])
##   user  system elapsed
##  1.317   0.000   1.319

stopifnot(identical(hv, h2),
          identical(is | FALSE, is != 0),
	  validObject(svN), validObject(lis), as.logical(is.na(svN[4])),
	  identical(is^2 > 0,	is & TRUE),
	  all(!lis), !any(lis), length(nn@i) == 0, !any(nn), all(!nn),
	  sum(lis) == 0, !prod(lis), range(lis) == c(0,0))

## create and use the t(.) method:
t(x20 <- sparseVector(c(9,3:1), i=c(1:2,4,7), length=20))
(T20 <- toeplitz(x20))
stopifnot(is(T20, "symmetricMatrix"), is(T20, "sparseMatrix"),
	  identical(unname(as.matrix(T20)),
                    toeplitz(as.vector(x20))))

## c() method for "sparseVector" - also available as regular function
(c1 <- c(x20, 0,0,0, -10*x20))
(c2 <- c(ns, is, FALSE))
(c3 <- c(ns, !ns, TRUE, NA, FALSE))
(c4 <- c(ns, rev(ns)))
## here, c() would produce a list {not dispatching to c.sparseVector()}
(c5 <- c.sparseVector(0,0, x20))

## checking (consistency)
.v <- as.vector
.s <- function(v) as(v, "sparseVector")
stopifnot(
  all.equal(c1, .s(c(.v(x20), 0,0,0, -10*.v(x20))),      tol=0),
  all.equal(c2, .s(c(.v(ns), .v(is), FALSE)),            tol=0),
  all.equal(c3, .s(c(.v(ns), !.v(ns), TRUE, NA, FALSE)), tol=0),
  all.equal(c4, .s(c(.v(ns), rev(.v(ns)))),              tol=0),
  all.equal(c5, .s(c(0,0, .v(x20))),                     tol=0)
)



cleanEx()
nameEx("sparseVector")
### * sparseVector

flush(stderr()); flush(stdout())

### Name: sparseVector
### Title: Sparse Vector Construction from Nonzero Entries
### Aliases: sparseVector
### Keywords: array

### ** Examples

str(sv <- sparseVector(x = 1:10, i = sample(999, 10), length=1000))

sx <- c(0,0,3, 3.2, 0,0,0,-3:1,0,0,2,0,0,5,0,0)
ss <- as(sx, "sparseVector")
stopifnot(identical(ss,
   sparseVector(x = c(2, -1, -2, 3, 1, -3, 5, 3.2),
                i = c(15L, 10:9, 3L,12L,8L,18L, 4L), length = 20L)))

(ns <- sparseVector(i= c(7, 3, 2), length = 10))
stopifnot(identical(ns,
      new("nsparseVector", length = 10, i = c(2, 3, 7))))



cleanEx()
nameEx("symmetricMatrix-class")
### * symmetricMatrix-class

flush(stderr()); flush(stdout())

### Name: symmetricMatrix-class
### Title: Virtual Class of Symmetric Matrices in Package Matrix
### Aliases: symmetricMatrix-class coerce,matrix,symmetricMatrix-method
###   dimnames,symmetricMatrix-method
### Keywords: classes

### ** Examples

## An example about the symmetric Dimnames:
sy <- sparseMatrix(i= c(2,4,3:5), j= c(4,7:5,5), x = 1:5, dims = c(7,7),
                   symmetric=TRUE, dimnames = list(NULL, letters[1:7]))
sy # shows symmetrical dimnames
sy@Dimnames  # internally only one part is stored
dimnames(sy) # both parts - as sy *is* symmetrical
## Don't show: 
local({ nm <- letters[1:7]
  stopifnot(identical(dimnames(sy), list(  nm, nm)),
	    identical(sy@Dimnames , list(NULL, nm)))
})
## End(Don't show)
showClass("symmetricMatrix")

## The names of direct subclasses:
scl <- getClass("symmetricMatrix")@subclasses
directly <- sapply(lapply(scl, slot, "by"), length) == 0
names(scl)[directly]

## Methods -- applicaple to all subclasses above:
showMethods(classes = "symmetricMatrix")



cleanEx()
nameEx("symmpart")
### * symmpart

flush(stderr()); flush(stdout())

### Name: symmpart
### Title: Symmetric Part and Skew(symmetric) Part of a Matrix
### Aliases: symmpart symmpart-methods skewpart skewpart-methods
###   symmpart,CsparseMatrix-method symmpart,RsparseMatrix-method
###   symmpart,TsparseMatrix-method symmpart,diagonalMatrix-method
###   symmpart,indMatrix-method symmpart,packedMatrix-method
###   symmpart,matrix-method symmpart,unpackedMatrix-method
###   skewpart,CsparseMatrix-method skewpart,RsparseMatrix-method
###   skewpart,TsparseMatrix-method skewpart,diagonalMatrix-method
###   skewpart,indMatrix-method skewpart,packedMatrix-method
###   skewpart,matrix-method skewpart,unpackedMatrix-method
### Keywords: array arith

### ** Examples

m <- Matrix(1:4, 2,2)
symmpart(m)
skewpart(m)

stopifnot(all(m == symmpart(m) + skewpart(m)))

dn <- dimnames(m) <- list(row = c("r1", "r2"), col = c("var.1", "var.2"))
stopifnot(all(m == symmpart(m) + skewpart(m)))
colnames(m) <- NULL
stopifnot(all(m == symmpart(m) + skewpart(m)))
dimnames(m) <- unname(dn)
stopifnot(all(m == symmpart(m) + skewpart(m)))


## investigate the current methods:
showMethods(skewpart, include = TRUE)



cleanEx()
nameEx("triangularMatrix-class")
### * triangularMatrix-class

flush(stderr()); flush(stdout())

### Name: triangularMatrix-class
### Title: Virtual Class of Triangular Matrices in Package Matrix
### Aliases: triangularMatrix-class
###   Arith,triangularMatrix,diagonalMatrix-method
###   Compare,triangularMatrix,diagonalMatrix-method
###   Logic,triangularMatrix,diagonalMatrix-method
###   coerce,matrix,triangularMatrix-method
###   determinant,triangularMatrix,logical-method
### Keywords: classes

### ** Examples

showClass("triangularMatrix")

## The names of direct subclasses:
scl <- getClass("triangularMatrix")@subclasses
directly <- sapply(lapply(scl, slot, "by"), length) == 0
names(scl)[directly]

(m <- matrix(c(5,1,0,3), 2))
as(m, "triangularMatrix")



cleanEx()
nameEx("uniqTsparse")
### * uniqTsparse

flush(stderr()); flush(stdout())

### Name: uniqTsparse
### Title: Unique (Sorted) TsparseMatrix Representations
### Aliases: uniqTsparse anyDuplicatedT
### Keywords: utilities classes

### ** Examples

example("dgTMatrix-class", echo=FALSE)
## -> 'T2'  with (i,j,x) slots of length 5 each
T2u <- uniqTsparse(T2)
stopifnot(## They "are" the same (and print the same):
          all.equal(T2, T2u, tol=0),
          ## but not internally:
          anyDuplicatedT(T2)  == 2,
          anyDuplicatedT(T2u) == 0,
          length(T2 @x) == 5,
          length(T2u@x) == 3)

## is 'x' a "uniq Tsparse" Matrix ?  [requires x to be TsparseMatrix!]
non_uniqT <- function(x, di = dim(x))
  is.unsorted(x@j) || anyDuplicatedT(x, di)
non_uniqT(T2 ) # TRUE
non_uniqT(T2u) # FALSE

T3 <- T2u
T3[1, c(1,3)] <- 10; T3[2, c(1,5)] <- 20
T3u <- uniqTsparse(T3)
str(T3u) # sorted in 'j', and within j, sorted in i
stopifnot(!non_uniqT(T3u))

## Logical l.TMatrix and n.TMatrix :
(L2 <- T2 > 0)
validObject(L2u <- uniqTsparse(L2))
(N2 <- as(L2, "nMatrix"))
validObject(N2u <- uniqTsparse(N2))
stopifnot(N2u@i == L2u@i, L2u@i == T2u@i,  N2@i == L2@i, L2@i == T2@i,
          N2u@j == L2u@j, L2u@j == T2u@j,  N2@j == L2@j, L2@j == T2@j)
# now with a nasty NA  [partly failed in Matrix 1.1-5]:
L.0N <- L.1N <- L2
L.0N@x[1:2] <- c(FALSE, NA)
L.1N@x[1:2] <- c(TRUE, NA)
validObject(L.0N)
validObject(L.1N)
(m.0N <- as.matrix(L.0N))
(m.1N <- as.matrix(L.1N))
stopifnot(identical(10L, which(is.na(m.0N))), !anyNA(m.1N))
symnum(m.0N)
symnum(m.1N)



cleanEx()
nameEx("unpack")
### * unpack

flush(stderr()); flush(stdout())

### Name: unpack
### Title: Representation of Packed and Unpacked Dense Matrices
### Aliases: pack unpack pack,dgeMatrix-method pack,lgeMatrix-method
###   pack,matrix-method pack,ngeMatrix-method pack,packedMatrix-method
###   pack,sparseMatrix-method pack,unpackedMatrix-method
###   unpack,matrix-method unpack,packedMatrix-method
###   unpack,sparseMatrix-method unpack,unpackedMatrix-method
### Keywords: array algebra

### ** Examples

showMethods("pack")
(s <- crossprod(matrix(sample(15), 5,3))) # traditional symmetric matrix
(sp <- pack(s))
mt <- as.matrix(tt <- tril(s))
(pt <- pack(mt))
stopifnot(identical(pt, pack(tt)),
	  dim(s ) == dim(sp), all(s  == sp),
	  dim(mt) == dim(pt), all(mt == pt), all(mt == tt))

showMethods("unpack")
(cp4 <- chol(Hilbert(4))) # is triangular
tp4 <- pack(cp4) # [t]riangular [p]acked
str(tp4)
(unpack(tp4))
stopifnot(identical(tp4, pack(unpack(tp4))))

z1 <- new("dsyMatrix", Dim = c(2L, 2L), x = as.double(1:4), uplo = "U")
z2 <- unpack(pack(z1))
stopifnot(!identical(z1, z2), # _not_ identical
          all(z1 == z2)) # but mathematically equal
cbind(z1@x, z2@x) # (unused!) lower triangle is "lost" in translation



cleanEx()
nameEx("unpackedMatrix-class")
### * unpackedMatrix-class

flush(stderr()); flush(stdout())

### Name: unpackedMatrix-class
### Title: Virtual Class '"unpackedMatrix"' of Unpacked Dense Matrices
### Aliases: unpackedMatrix-class coerce,matrix,unpackedMatrix-method
###   coerce,numLike,unpackedMatrix-method diag,unpackedMatrix-method
###   diag<-,unpackedMatrix-method t,unpackedMatrix-method
### Keywords: classes

### ** Examples

showClass("unpackedMatrix")
showMethods(classes = "unpackedMatrix")



cleanEx()
nameEx("unused-classes")
### * unused-classes

flush(stderr()); flush(stdout())

### Name: Unused-classes
### Title: Virtual Classes Not Yet Really Implemented and Used
### Aliases: iMatrix-class zMatrix-class
### Keywords: classes

### ** Examples

showClass("iMatrix")
showClass("zMatrix")



cleanEx()
nameEx("updown")
### * updown

flush(stderr()); flush(stdout())

### Name: updown
### Title: Up- and Down-Dating a Cholesky Decomposition
### Aliases: updown updown-methods updown,ANY,ANY,ANY-method
###   updown,character,mMatrix,CHMfactor-method
###   updown,logical,mMatrix,CHMfactor-method
### Keywords: methods

### ** Examples

dn <- list(LETTERS[1:3], letters[1:5])
## pointer vectors can be used, and the (i,x) slots are sorted if necessary:
m <- sparseMatrix(i = c(3,1, 3:2, 2:1), p= c(0:2, 4,4,6), x = 1:6, dimnames = dn)
cA <- Cholesky(A <- crossprod(m) + Diagonal(5))
166 * as(cA,"Matrix") ^ 2
uc1 <- updown("+",   Diagonal(5), cA)
## Hmm: this loses positive definiteness:
uc2 <- updown("-", 2*Diagonal(5), cA)
image(show(as(cA, "Matrix")))
image(show(c2 <- as(uc2,"Matrix")))# severely negative entries
##--> Warning



cleanEx()
nameEx("wrld_1deg")
### * wrld_1deg

flush(stderr()); flush(stdout())

### Name: wrld_1deg
### Title: World 1-degree grid contiguity matrix
### Aliases: wrld_1deg
### Keywords: datasets

### ** Examples

data(wrld_1deg)
(n <- ncol(wrld_1deg))
IM <- .symDiagonal(n)
doExtras <- interactive() || nzchar(Sys.getenv("R_MATRIX_CHECK_EXTRA"))
nn <- if(doExtras) 20 else 3
set.seed(1)
rho <- runif(nn, 0, 1)
system.time(MJ <- sapply(rho,
                   function(x) determinant(IM - x * wrld_1deg,
                                           logarithm = TRUE)$modulus))
nWC <- -wrld_1deg
C1 <- Cholesky(nWC, Imult = 2)
## Note that det(<CHMfactor>) = det(L) = sqrt(det(A))
## ====> log det(A) = log( det(L)^2 ) = 2 * log det(L) :
system.time(MJ1 <- n * log(rho) +
   sapply(rho, function(x) c(2* determinant(update(C1, nWC, 1/x))$modulus))
)
stopifnot(all.equal(MJ, MJ1))
C2 <- Cholesky(nWC, super = TRUE, Imult = 2)
system.time(MJ2 <- n * log(rho) +
   sapply(rho, function(x) c(2* determinant(update(C2, nWC, 1/x))$modulus))
)
system.time(MJ3 <- n * log(rho) + Matrix:::ldetL2up(C1, nWC, 1/rho))
system.time(MJ4 <- n * log(rho) + Matrix:::ldetL2up(C2, nWC, 1/rho))
stopifnot(all.equal(MJ, MJ2),
          all.equal(MJ, MJ3),
          all.equal(MJ, MJ4))



### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
