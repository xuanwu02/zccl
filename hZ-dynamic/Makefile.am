##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

#AM_CFLAGS = -I./include -I../zlib
#LDFLAGS=-fPIC -shared

AUTOMAKE_OPTIONS=foreign
include_HEADERS=include/ZCCL.h include/ZCCL_rw.h include/ZCCL_defines.h include/ZCCL_BytesToolkit.h include/ZCCL_TypeManager.h include/ZCCL_dataCompression.h include/ZCCL_utility.h include/ZCCL_float.h include/ZCCLd_float.h include/ZCCL_helper.h

lib_LTLIBRARIES=libZCCL.la
libZCCL_la_CFLAGS=-I./include
if OPENMP
libZCCL_la_CFLAGS+=-fopenmp
endif
libZCCL_la_LDFLAGS = -version-info  1:0:0
libZCCL_la_SOURCES=src/ZCCL.c src/ZCCL_rw.c src/ZCCL_BytesToolkit.c src/ZCCL_TypeManager.c src/ZCCL_dataCompression.c src/ZCCL_utility.c src/ZCCL_float.c src/ZCCLd_float.c src/ZCCL_helper.c

libSZ_la_LINK= $(AM_V_CC)$(LIBTOOL) --tag=CC --mode=link $(CCLD) $(libZCCL_la_CFLAGS) -O3 $(libZCCL_la_LDFLAGS) -o $(lib_LTLIBRARIES)
