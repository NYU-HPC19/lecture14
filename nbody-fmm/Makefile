
PVFMM_DIR=./pvfmm
-include $(PVFMM_DIR)/MakeVariables

ifndef CXXFLAGS_PVFMM
$(error Cannot find file: MakeVariables)
endif

CXX=$(CXX_PVFMM)
CXXFLAGS=$(CXXFLAGS_PVFMM)
LDLIBS=$(LDLIBS_PVFMM)

TARGETS = $(basename $(wildcard *.cpp))

all : $(TARGETS)

%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@

clean:
	-$(RM) $(TARGETS) *vtu *~

.PHONY: all, clean
