##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

AUTOMAKE_OPTIONS=foreign
AM_CFLAGS = -I../hZ-dynamic/include
AM_FCFLAGS = -I../hZ-dynamic/include -I../hZ-dynamic
if OPENMP
AM_CFLAGS+=-fopenmp
endif


bin_PROGRAMS=testfloat_homocompress_fastmode1_arg compression_allreduce

testfloat_homocompress_fastmode1_arg_SOURCES=testfloat_homocompress_fastmode1_arg.c
testfloat_homocompress_fastmode1_arg_LDADD=../hZ-dynamic/.libs/libhZCCL.a -lm

compression_allreduce_SOURCES=compression_allreduce.c
compression_allreduce_LDADD=../hZ-dynamic/.libs/libhZCCL.a -lm
