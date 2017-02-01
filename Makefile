# VR CVUT segmenter makefile

all:	segmenter
segmenter:	main.cpp Segmenter.cpp
	@echo
	@echo VR CVUT Segmenter building.
	@echo
	g++ -O -Wall -o segmenter main.cpp Segmenter.cpp -Ivr -Lvr -lvr -lm -ldl
	@echo Compilation successfully finished.
	@echo

clean:
	rm -rf *.o
