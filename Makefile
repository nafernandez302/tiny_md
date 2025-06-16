CC      = clang # activar compilador AMD (!)
CFLAGS	= -O3 -ffast-math -march=native
CPPFLAGS = -DN=$(N)
WFLAGS	= -std=c11 -Wall -Wextra -g
LDFLAGS	= -lm
N?=256

TARGETS	= tiny_md
SOURCES	= $(shell echo *.c)
OBJECTS = core.o wtime.o

all: $(TARGETS)

viz: viz.o $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lGL -lGLU -lglut

tiny_md: tiny_md.o $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(WFLAGS) $(CPPFLAGS) $(CFLAGS) -c $<

clean:
	rm -f $(TARGETS) *.o *.xyz *.log .depend

.depend: $(SOURCES)
	$(CC) -MM $^ > $@

-include .depend

.PHONY: clean all
