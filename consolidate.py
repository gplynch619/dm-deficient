import os
import sys
import numpy as np

directory = sys.argv[1]

files = os.listdir(directory)

iterfiles = iter(files)

anchor = np.load(os.path.join(directory, next(iterfiles)))

print anchor

for filename in iterfiles:
	a = np.load(os.path.join(directory,filename))
	print a
	anchor = np.concatenate((anchor, a))
	
np.save(os.path.join(directory, sys.argv[2]), anchor)

