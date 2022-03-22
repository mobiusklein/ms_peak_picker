test:
	py.test -v ms_peak_picker --cov=ms_peak_picker --cov-report=html --cov-report term

retest:
	py.test -v --pdb ms_peak_picker --lf

sphinx:
	cd docs && make html

dev:
	python setup.py develop
