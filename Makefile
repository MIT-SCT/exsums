# Copyright (c) 2019 MIT License by Supertech
CC := clang++

SRC1 := test/testbed_exsum.cpp
SRC2 := test/testbed_exsum_memory.cpp


SRCS := $(SRC1) $(SRC2)

ARCH := native

# Set the name of your binary.  Change it if you like.
PRODUCT1 := exsum
PRODUCT2 := exsum_memory

PRODUCTS := $(PRODUCT1) $(PRODUCT2)

ifdef BOOST_ROOT
BOOST_INCLUDE := -I $(BOOST_ROOT)
else
BOOST_INCLUDE :=
endif

INCLUDE = -I./include $(BOOST_INCLUDE)
LIBS = -lgtest -lgtest_main -lpthread

UNAME := $(shell uname)

# These flags will be applied to your code any time it is built.
CFLAGS := -Wall -std=c++17 -mavx2 $(INCLUDE) -mcx16

# These flags are applied only if you build your code with "make DEBUG=1".  -g
# generates debugging symbols, -DDEBUG defines the preprocessor symbol "DEBUG"
# (so that you can use "#ifdef DEBUG" in your code), and -O0 disables compiler
# optimizations, so that the binary generated more directly corresponds to your
# source code.
CFLAGS_DEBUG := -g -DDEBUG -O0

# In the release version, we ask for many optimizations; -O3 sets the
# optimization level to three.  -DNDEBUG defines the NDEBUG macro,
# which disables assertion checks.
CFLAGS_RELEASE := -O3 -DNDEBUG -march=$(ARCH) -flto

# These flags are used to invoke Clang's address sanitizer.
CFLAGS_ASAN := -O1 -g -fsanitize=address

# These flags are applied when linking object files together into your binary.
# If you need to link against libraries, add the appropriate flags here.  By
# default, your code is linked against the "rt" library with the flag -lrt;
# this library is used by the timing code in the testbed.
ifeq ($(UNAME),Darwin)
  CFLAGS += -DMACPORT
  LDFLAGS := -flto $(LIBS)
else
  LDFLAGS := -lrt -flto -fuse-ld=gold $(LIBS)
endif
ifeq ($(PARALLEL),1)
  CFLAGS += -fcilkplus -DPARALLEL -DCILK
  LDFLAGS += -lcilkrts
endif

# set DELAY if not defined
ifeq ($(DELAY),)
DELAY := 0
endif

ifeq ($(shell test $(DELAY) -gt 0; echo $$?),0)
  CFLAGS += -DDELAY=$(DELAY)
endif

################################################################################
# You probably won't need to change anything below this line, but if you're
# curious about how makefiles work, or if you'd like to customize the behavior
# of your makefile, go ahead and take a peek.
################################################################################

# You shouldn't need to touch this.  This keeps track of whether you are
# building in a release or debug configuration, and sets CFLAGS appropriately.
# (This mechanism is based on one from the original materials for 6.197 by
# Ceryen Tan and Marek Olszewski.)
OLDMODE=$(shell cat .buildmode 2> /dev/null)
ifeq ($(DEBUG),1)
  CFLAGS := $(CFLAGS_DEBUG) $(CFLAGS)
  ifneq ($(OLDMODE),debug)
    $(shell echo debug > .buildmode)
  endif
else ifeq ($(ASAN),1)
  CFLAGS := $(CFLAGS_ASAN) $(CFLAGS)
  LDFLAGS := $(LDFLAGS) -fsanitize=address
  ifneq ($(OLDMODE),asan)
    $(shell echo asan > .buildmode)
  endif
else
  CFLAGS := $(CFLAGS_RELEASE) $(CFLAGS)
  ifneq ($(OLDMODE),nodebug)
    $(shell echo nodebug > .buildmode)
  endif
endif

# When you invoke make without an argument, make behaves as though you had
# typed "make all", and builds whatever you have listed here.  (It knows to
# pick "make all" because "all" is the first rule listed.)
.PHONY: all
all: $(PRODUCTS)

# This special "target" will remove the binary and all intermediate files.
.PHONY: clean
clean:
	rm -f $(OBJS) $(PRODUCTS) \
        $(addsuffix .gcda, $(basename $(SRCS))) \
        $(addsuffix .gcno, $(basename $(SRCS))) \
        $(addsuffix .gcov, $(SRCS) fasttime.h)

# This rule generates a list of object names.  Each of your source files (but
# not your header files) produces a single object file when it's compiled.  In
# a later step, all of those object files are linked together to produce the
# binary that you run.
OBJ1 = $(addsuffix .o, $(basename $(SRC1)))
OBJ2 = $(addsuffix .o, $(basename $(SRC2)))
OBJ3 = $(addsuffix .o, $(basename $(SRC3)))

OBJS = $(OBJ1) $(OBJ2) $(OBJ3)

# These rules tell make how to automatically generate rules that build the
# appropriate object-file from each of the source files listed in SRC (above).
%.o : %.c .buildmode
	$(CC) $(INCLUDE) $(CFLAGS) -c $< -o $@
%.o : %.cc .buildmode
	$(CC) $(INCLUDE) $(CFLAGS) -c $< -o $@
%.o : %.cpp .buildmode
	$(CC) $(INCLUDE) $(CFLAGS) -c $< -o $@

# This rule tells make that it can produce your binary by linking together all
# of the object files produced from your source files and any necessary
# libraries.
$(PRODUCT1): $(OBJ1) .buildmode
	$(CC) $(INCLUDE) -o $@ $(OBJ1) $(LDFLAGS)
$(PRODUCT2): $(OBJ2) .buildmode
	$(CC) $(INCLUDE) -o $@ $(OBJ2) $(LDFLAGS)
$(PRODUCT3): $(OBJ3) .buildmode
	$(CC) $(INCLUDE) -o $@ $(OBJ3) $(LDFLAGS)
