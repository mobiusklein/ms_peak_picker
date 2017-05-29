test:
	py.test -v  ms_peak_picker --cov=ms_peak_picker --cov-report=html

retest:
	py.test -v ms_peak_picker --lf