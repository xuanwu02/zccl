##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

AUTOMAKE_OPTIONS=foreign
AM_CFLAGS = -I../ZCCL/include
AM_FCFLAGS = -I../ZCCL/include -I../ZCCL
if OPENMP
AM_CFLAGS+=-fopenmp
endif


bin_PROGRAMS=testfloat_homocompress_fastmode1_arg compression_allreduce test_allgv_3d

testfloat_homocompress_fastmode1_arg_SOURCES=testfloat_homocompress_fastmode1_arg.c
testfloat_homocompress_fastmode1_arg_LDADD=../ZCCL/.libs/libZCCL.a -lm

compression_allreduce_SOURCES=compression_allreduce.c
compression_allreduce_LDADD=../ZCCL/.libs/libZCCL.a -lm

test_allgv_3d_SOURCES=test_allgv_3d.c
test_allgv_3d_LDADD=../ZCCL/.libs/libZCCL.a -lm
