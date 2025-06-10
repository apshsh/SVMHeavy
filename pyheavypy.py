def convgen(h):
    import pyheavy
    if h is None:
        #print("N")
        pyheavy.z_compush("N")
    elif type(h) is int:
        #print("Z")
        pyheavy.z_compush("Z")
        pyheavy.z_intpush(h)
    elif type(h) is float:
        #print("R")
        pyheavy.z_compush("R")
        pyheavy.z_dblpush(h)
    elif type(h) is complex:
        #print("C")
        pyheavy.z_compush("C")
        pyheavy.z_dblpush(h.real)
        pyheavy.z_dblpush(h.imag)
    elif type(h) is str:
        #print("S")
        pyheavy.z_compush("S")
        pyheavy.z_strpush(h)
    elif type(h) is list:
        #print("V")
        pyheavy.z_compush("V")
        pyheavy.z_intpush(len(h))
        for x in h:
            convgen(x)
    elif type(h) is tuple:
        #print("X")
        pyheavy.z_compush("X")
        pyheavy.z_intpush(len(h))
        for x in h:
            convgen(x)
    else:
        #print("E")
        pyheavy.z_compush("E")

