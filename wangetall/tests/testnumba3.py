import numpy as np
from scipy import stats


print(stats.chi2.ppf(0.95, 5))
print(stats.distributions.chi2.sf(0.95, 5))
