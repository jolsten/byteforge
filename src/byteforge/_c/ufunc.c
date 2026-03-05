#include <Python.h>
#include "./utils.c"
#include "./bcd.c"
#include "./gray.c"
#include "./milstd1750a.c"
#include "./ti.c"
#include "./ibm.c"
#include "./dec.c"

static PyMethodDef Methods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ufunc",
    NULL,
    -1,
    Methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_ufunc(void)
{
    PyObject *m, *d;

    import_array();
    import_umath();

    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    d = PyModule_GetDict(m);

    /* BCD */
    PyObject *bcd_decode = PyUFunc_FromFuncAndData(
        bcd_decode_funcs, NULL, bcd_decode_types, 1, 2, 1,
        PyUFunc_None, "bcd_decode", "BCD decode ufunc", 0
    );
    PyDict_SetItemString(d, "bcd_decode", bcd_decode);
    Py_DECREF(bcd_decode);

    PyObject *bcd_encode = PyUFunc_FromFuncAndData(
        bcd_encode_funcs, NULL, bcd_encode_types, 1, 2, 1,
        PyUFunc_None, "bcd_encode", "BCD encode ufunc", 0
    );
    PyDict_SetItemString(d, "bcd_encode", bcd_encode);
    Py_DECREF(bcd_encode);

    /* GrayCode */
    PyObject *gray_decode = PyUFunc_FromFuncAndData(
        gray_decode_funcs, NULL, gray_decode_types, 1, 2, 1,
        PyUFunc_None, "gray_decode", "Gray code decode ufunc", 0
    );
    PyDict_SetItemString(d, "gray_decode", gray_decode);
    Py_DECREF(gray_decode);

    /* MilStd1750A */
    PyObject *milstd1750a32_decode = PyUFunc_FromFuncAndData(
        milstd1750a32_decode_funcs, NULL, milstd1750a32_decode_types, 1, 1, 1,
        PyUFunc_None, "milstd1750a32_decode", "MIL-STD-1750A 32-bit decode ufunc", 0
    );
    PyDict_SetItemString(d, "milstd1750a32_decode", milstd1750a32_decode);
    Py_DECREF(milstd1750a32_decode);

    PyObject *milstd1750a48_decode = PyUFunc_FromFuncAndData(
        milstd1750a48_decode_funcs, NULL, milstd1750a48_decode_types, 1, 1, 1,
        PyUFunc_None, "milstd1750a48_decode", "MIL-STD-1750A 48-bit decode ufunc", 0
    );
    PyDict_SetItemString(d, "milstd1750a48_decode", milstd1750a48_decode);
    Py_DECREF(milstd1750a48_decode);

    PyObject *milstd1750a32_encode = PyUFunc_FromFuncAndData(
        milstd1750a32_encode_funcs, NULL, milstd1750a32_encode_types, 1, 1, 1,
        PyUFunc_None, "milstd1750a32_encode", "MIL-STD-1750A 32-bit encode ufunc", 0
    );
    PyDict_SetItemString(d, "milstd1750a32_encode", milstd1750a32_encode);
    Py_DECREF(milstd1750a32_encode);

    PyObject *milstd1750a48_encode = PyUFunc_FromFuncAndData(
        milstd1750a48_encode_funcs, NULL, milstd1750a48_encode_types, 1, 1, 1,
        PyUFunc_None, "milstd1750a48_encode", "MIL-STD-1750A 48-bit encode ufunc", 0
    );
    PyDict_SetItemString(d, "milstd1750a48_encode", milstd1750a48_encode);
    Py_DECREF(milstd1750a48_encode);

    /* TIFloat */
    PyObject *ti32_decode = PyUFunc_FromFuncAndData(
        ti32_decode_funcs, NULL, ti32_decode_types, 1, 1, 1,
        PyUFunc_None, "ti32_decode", "TI 32-bit decode ufunc", 0
    );
    PyDict_SetItemString(d, "ti32_decode", ti32_decode);
    Py_DECREF(ti32_decode);

    PyObject *ti40_decode = PyUFunc_FromFuncAndData(
        ti40_decode_funcs, NULL, ti40_decode_types, 1, 1, 1,
        PyUFunc_None, "ti40_decode", "TI 40-bit decode ufunc", 0
    );
    PyDict_SetItemString(d, "ti40_decode", ti40_decode);
    Py_DECREF(ti40_decode);

    PyObject *ti32_encode = PyUFunc_FromFuncAndData(
        ti32_encode_funcs, NULL, ti32_encode_types, 1, 1, 1,
        PyUFunc_None, "ti32_encode", "TI 32-bit encode ufunc", 0
    );
    PyDict_SetItemString(d, "ti32_encode", ti32_encode);
    Py_DECREF(ti32_encode);

    PyObject *ti40_encode = PyUFunc_FromFuncAndData(
        ti40_encode_funcs, NULL, ti40_encode_types, 1, 1, 1,
        PyUFunc_None, "ti40_encode", "TI 40-bit encode ufunc", 0
    );
    PyDict_SetItemString(d, "ti40_encode", ti40_encode);
    Py_DECREF(ti40_encode);

    /* IBMFloat */
    PyObject *ibm32_decode = PyUFunc_FromFuncAndData(
        ibm32_decode_funcs, NULL, ibm32_decode_types, 1, 1, 1,
        PyUFunc_None, "ibm32_decode", "IBM 32-bit decode ufunc", 0
    );
    PyDict_SetItemString(d, "ibm32_decode", ibm32_decode);
    Py_DECREF(ibm32_decode);

    PyObject *ibm64_decode = PyUFunc_FromFuncAndData(
        ibm64_decode_funcs, NULL, ibm64_decode_types, 1, 1, 1,
        PyUFunc_None, "ibm64_decode", "IBM 64-bit decode ufunc", 0
    );
    PyDict_SetItemString(d, "ibm64_decode", ibm64_decode);
    Py_DECREF(ibm64_decode);

    PyObject *ibm32_encode = PyUFunc_FromFuncAndData(
        ibm32_encode_funcs, NULL, ibm32_encode_types, 1, 1, 1,
        PyUFunc_None, "ibm32_encode", "IBM 32-bit encode ufunc", 0
    );
    PyDict_SetItemString(d, "ibm32_encode", ibm32_encode);
    Py_DECREF(ibm32_encode);

    PyObject *ibm64_encode = PyUFunc_FromFuncAndData(
        ibm64_encode_funcs, NULL, ibm64_encode_types, 1, 1, 1,
        PyUFunc_None, "ibm64_encode", "IBM 64-bit encode ufunc", 0
    );
    PyDict_SetItemString(d, "ibm64_encode", ibm64_encode);
    Py_DECREF(ibm64_encode);

    /* DECFloat */
    PyObject *dec32_decode = PyUFunc_FromFuncAndData(
        dec32_decode_funcs, NULL, dec32_decode_types, 1, 1, 1,
        PyUFunc_None, "dec32_decode", "DEC F4 32-bit decode ufunc", 0
    );
    PyDict_SetItemString(d, "dec32_decode", dec32_decode);
    Py_DECREF(dec32_decode);

    PyObject *dec64_decode = PyUFunc_FromFuncAndData(
        dec64_decode_funcs, NULL, dec64_decode_types, 1, 1, 1,
        PyUFunc_None, "dec64_decode", "DEC D4 64-bit decode ufunc", 0
    );
    PyDict_SetItemString(d, "dec64_decode", dec64_decode);
    Py_DECREF(dec64_decode);

    PyObject *dec64g_decode = PyUFunc_FromFuncAndData(
        dec64g_decode_funcs, NULL, dec64g_decode_types, 1, 1, 1,
        PyUFunc_None, "dec64g_decode", "DEC G4 64-bit decode ufunc", 0
    );
    PyDict_SetItemString(d, "dec64g_decode", dec64g_decode);
    Py_DECREF(dec64g_decode);

    PyObject *dec32_encode = PyUFunc_FromFuncAndData(
        dec32_encode_funcs, NULL, dec32_encode_types, 1, 1, 1,
        PyUFunc_None, "dec32_encode", "DEC F4 32-bit encode ufunc", 0
    );
    PyDict_SetItemString(d, "dec32_encode", dec32_encode);
    Py_DECREF(dec32_encode);

    PyObject *dec64_encode = PyUFunc_FromFuncAndData(
        dec64_encode_funcs, NULL, dec64_encode_types, 1, 1, 1,
        PyUFunc_None, "dec64_encode", "DEC D4 64-bit encode ufunc", 0
    );
    PyDict_SetItemString(d, "dec64_encode", dec64_encode);
    Py_DECREF(dec64_encode);

    PyObject *dec64g_encode = PyUFunc_FromFuncAndData(
        dec64g_encode_funcs, NULL, dec64g_encode_types, 1, 1, 1,
        PyUFunc_None, "dec64g_encode", "DEC G4 64-bit encode ufunc", 0
    );
    PyDict_SetItemString(d, "dec64g_encode", dec64g_encode);
    Py_DECREF(dec64g_encode);

    return m;
}
