#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>


static PyObject *accumulate_distance(PyObject *self, PyObject *args) {
    PyObject *distances_object;
    int band;
    Py_buffer distances;
    double *previous_cost = NULL;
    double *current_cost = NULL;
    int32_t *previous_steps = NULL;
    int32_t *current_steps = NULL;
    double result = INFINITY;

    (void)self;
    if (!PyArg_ParseTuple(args, "Oi:accumulate_distance", &distances_object, &band)) {
        return NULL;
    }
    if (band < 0) {
        PyErr_SetString(PyExc_ValueError, "band must be non-negative");
        return NULL;
    }
    if (PyObject_GetBuffer(
            distances_object,
            &distances,
            PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES
        ) < 0) {
        return NULL;
    }
    if (
        distances.ndim != 2 || distances.itemsize != (Py_ssize_t)sizeof(float) ||
        distances.format == NULL || distances.format[0] != 'f' ||
        distances.shape[0] < 1 || distances.shape[1] < 1
    ) {
        PyBuffer_Release(&distances);
        PyErr_SetString(
            PyExc_ValueError,
            "distances must be a non-empty two-dimensional float32 array"
        );
        return NULL;
    }

    const Py_ssize_t rows = distances.shape[0];
    const Py_ssize_t columns = distances.shape[1];
    const size_t cost_bytes = (size_t)(columns + 1) * sizeof(double);
    const size_t step_bytes = (size_t)(columns + 1) * sizeof(int32_t);
    previous_cost = (double *)malloc(cost_bytes);
    current_cost = (double *)malloc(cost_bytes);
    previous_steps = (int32_t *)malloc(step_bytes);
    current_steps = (int32_t *)malloc(step_bytes);
    if (
        previous_cost == NULL || current_cost == NULL ||
        previous_steps == NULL || current_steps == NULL
    ) {
        free(previous_cost);
        free(current_cost);
        free(previous_steps);
        free(current_steps);
        PyBuffer_Release(&distances);
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS

    for (Py_ssize_t column = 0; column <= columns; ++column) {
        previous_cost[column] = INFINITY;
        previous_steps[column] = 0;
    }
    previous_cost[0] = 0.0;

    for (Py_ssize_t row = 1; row <= rows; ++row) {
        for (Py_ssize_t column = 0; column <= columns; ++column) {
            current_cost[column] = INFINITY;
            current_steps[column] = 0;
        }
        const Py_ssize_t start = row - band > 1 ? row - band : 1;
        const Py_ssize_t end = row + band < columns ? row + band : columns;

        for (Py_ssize_t column = start; column <= end; ++column) {
            /* Preserve the Python reference's up, left, diagonal tie order. */
            double best_cost = previous_cost[column];
            int32_t best_steps = previous_steps[column];
            if (current_cost[column - 1] < best_cost) {
                best_cost = current_cost[column - 1];
                best_steps = current_steps[column - 1];
            }
            if (previous_cost[column - 1] < best_cost) {
                best_cost = previous_cost[column - 1];
                best_steps = previous_steps[column - 1];
            }

            const char *cell =
                (const char *)distances.buf +
                (row - 1) * distances.strides[0] +
                (column - 1) * distances.strides[1];
            current_cost[column] = best_cost + (double)(*(const float *)cell);
            current_steps[column] = best_steps + 1;
        }

        double *cost_swap = previous_cost;
        previous_cost = current_cost;
        current_cost = cost_swap;
        int32_t *step_swap = previous_steps;
        previous_steps = current_steps;
        current_steps = step_swap;
    }

    if (isfinite(previous_cost[columns]) && previous_steps[columns] > 0) {
        result = previous_cost[columns] / (double)previous_steps[columns];
    }

    Py_END_ALLOW_THREADS

    free(previous_cost);
    free(current_cost);
    free(previous_steps);
    free(current_steps);
    PyBuffer_Release(&distances);
    return PyFloat_FromDouble(result);
}


static PyMethodDef module_methods[] = {
    {
        "accumulate_distance",
        accumulate_distance,
        METH_VARARGS,
        "Accumulate a path-length-normalized DTW distance matrix."
    },
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "_dtw_native",
    "Native DTW recurrence for the voice template matcher.",
    -1,
    module_methods,
};


PyMODINIT_FUNC PyInit__dtw_native(void) {
    return PyModule_Create(&module_definition);
}
