# ARGS: expected -g or -O3
# Usage: make ARGS=-g
CXXFLAGS  = -std=c++11 -Wall -Wextra -pedantic -lpthread -fopenmp ${ARGS}
CCXX = g++
OBJS = o/main.o o/model.o o/oneclass_svm.o o/dcdistance_occ.o o/heif.o o/ecoocc.o o/nbpc.o o/tfidf_vectorizer.o o/math_util.o o/io.o

all: main

main: $(OBJS)
	$(CCXX) -o $@ $^ -larmadillo -lmlpack $(CXXFLAGS)

o/main.o: main.cc
	$(CCXX) -c $< -o $@ $(CXXFLAGS)

o/model.o: src/model.cc
	$(CCXX) -c $< -o $@ $(CXXFLAGS)

o/oneclass_svm.o: src/oneclass_svm.cc
	$(CCXX) -c $< -o $@ $(CXXFLAGS)

o/dcdistance_occ.o: src/dcdistance_occ.cc
	$(CCXX) -c $< -o $@ $(CXXFLAGS)

o/nbpc.o: src/nbpc.cc
	$(CCXX) -c $< -o $@ $(CXXFLAGS)

o/ecoocc.o: src/ecoocc.cc
	$(CCXX) -c $< -o $@ $(CXXFLAGS)

o/heif.o: src/heif.cc
	$(CCXX) -c $< -o $@ $(CXXFLAGS)

o/math_util.o: src/math_util.cc
	$(CCXX) -c $< -o $@ $(CXXFLAGS)

o/io.o: src/io.cc
	$(CCXX) -c $< -o $@ $(CXXFLAGS)

o/tfidf_vectorizer.o: src/tfidf_vectorizer.cc
	$(CCXX) -c $< -o $@ $(CXXFLAGS)

clean:
	rm o/* main
