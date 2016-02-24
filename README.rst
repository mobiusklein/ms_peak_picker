A Library for Mass Spectrometry Peak Picking
--------------------------------------------

This is a small library to provide peak picking for software processing mass spectrometry data. It
is intended to solve only the peak picking problem, leaving the whole process of deisotoping and charge
state deconvolution for separate software.

The algorithm implemented here is primarily a direct translation of Navdeep Jaitly's C++ for the DeconEngineV2
component of PNNL's Decon2LS. It was not possible to simply wrap said library in a cross-platform fashion due to
it's dependence on CLR Managed C++, a deprecated feature of Microsoft Visual C++ which allowed MSVC to put native
objects' lifespan under the control of a garbage collector integrated with the CLR.

Critical paths are written using a statically typed C extension which depends upon Numpy. This requirement may be
waived in the future as Cython Typed Arrays may prove more scalable.



API
---


.. code:: python
    
    from ms_peak_picker import pick_peaks

    mz_array, intensity_array = get_spectrum_as_numpy_arrays()

    peak_list = pick_peaks(mz_array, intensity_array, fit_type="quadratic")

    peak = peak_list[0]
    print(peak.mz, peak.intensity, peak.full_width_at_half_max, peak.signal_to_noise)

    if peak_list.has_peak(204.08):
        print("Found that peak")
