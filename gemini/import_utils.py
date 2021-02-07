from sys import version_info

# unify functions
if version_info[0] == 2:
    from itertools import izip as zip, imap as map, ifilter as filter
filter = filter
map = map
zip = zip
if version_info[0] == 3:
    from functools import reduce
reduce = reduce
range = xrange if version_info[0] == 2 else  range

# unify itertools and functools
if version_info[0] == 2:
    from itertools import ifilterfalse as filterfalse
    from itertools import izip_longest as zip_longest
    from functools32 import partial, wraps
else:
    from itertools import filterfalse
    from itertools import zip_longest
    from functools import partial, wraps
