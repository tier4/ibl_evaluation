import numpy as np
import transformations
import re


R = transformations.quaternion_matrix([0.468120, 0.442822, -0.518901, 0.561706])
C = np.array([832.043051, -460.571622, 41.813090])

t = -R[:3, :3] @ C
print(t)

img_name = 'img_03359_c0_1303398766647625us.jpg'
print(re.findall('\d+', img_name))
