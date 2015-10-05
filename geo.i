%module DTgeo
%include <argcargv.i>

%apply (int ARGC, char **ARGV) { (int argc, char *argv[]) }
%{
extern int Tsteps;
int _main(int argc, char* argv[]);
#include "py_consts.h"
%}

%typemap(in) (int argc, char *argv[]) {
  int i;
  if (!PyList_Check($input)) {
    PyErr_SetString(PyExc_ValueError, "Expecting a list");
    return NULL;
  }
  $1 = PyList_Size($input);
  $2 = (char **) malloc(($1+1)*sizeof(char *));
  for (i = 0; i < $1; i++) {
    PyObject *s = PyList_GetItem($input,i);
    if (!PyString_Check(s)) {
      free($2);
      PyErr_SetString(PyExc_ValueError, "List items must be strings");
      return NULL;
    }
    $2[i] = PyString_AsString(s);
  }
  $2[i] = 0;
}
%typemap(freearg) (int argc, char *argv[]) {
  free($2); // If here is uneeded, free(NULL) is legal
}
extern int Tsteps;

%include "py_consts.h"
int _main(int argc, char* argv[]);
