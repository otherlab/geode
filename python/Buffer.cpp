//#####################################################################
// Class Buffer
//#####################################################################
#include <other/core/python/Buffer.h>
using namespace other;

#ifdef OTHER_PYTHON

PyTypeObject Buffer::pytype = {
    PyObject_HEAD_INIT(&PyType_Type)
    0,                          // ob_size
    "other.Buffer",             // tp_name
    sizeof(Buffer),             // tp_basicsize
    0,                          // tp_itemsize
    (destructor)free,           // tp_dealloc
    0,                          // tp_print
    0,                          // tp_getattr
    0,                          // tp_setattr
    0,                          // tp_compare
    0,                          // tp_repr
    0,                          // tp_as_number
    0,                          // tp_as_sequence
    0,                          // tp_as_mapping
    0,                          // tp_hash 
    0,                          // tp_call
    0,                          // tp_str
    0,                          // tp_getattro
    0,                          // tp_setattro
    0,                          // tp_as_buffer
    Py_TPFLAGS_DEFAULT,         // tp_flags
    "Raw memory buffer",        // tp_doc
    0,                          // tp_traverse
    0,                          // tp_clear
    0,                          // tp_richcompare
    0,                          // tp_weaklistoffset
    0,                          // tp_iter
    0,                          // tp_iternext
    0,                          // tp_methods
    0,                          // tp_members
    0,                          // tp_getset
    0,                          // tp_base
    0,                          // tp_dict
    0,                          // tp_descr_get
    0,                          // tp_descr_set
    0,                          // tp_dictoffset
    0,                          // tp_init
    0,                          // tp_alloc
    0,                          // tp_new
    0,                          // tp_free
};

// All necessary Buffer::pytpe fields are filled in, so no PyType_Ready is needed

#else // non-python stub

PyTypeObject Buffer::pytype = {
  "other.Buffer",           // tp_name
  (void(*)(PyObject*))free, // tp_dealloc
};

#endif
