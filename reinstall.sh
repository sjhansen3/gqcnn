pip uninstall gqcnn
rm -r -f build
rm -r -f dist
rm -r -f gqcnn.egg-info
python setup.py build
python setup.py install