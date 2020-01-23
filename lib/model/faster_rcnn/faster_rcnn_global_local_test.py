Traceback (most recent call last):
  File "/storage/lxdeng/.pyenv/versions/3.6.7/bin/autopep8", line 11, in <module>
    load_entry_point('autopep8==1.4.4', 'console_scripts', 'autopep8')()
  File "/storage/lxdeng/.pyenv/versions/3.6.7/lib/python3.6/site-packages/autopep8.py", line 4192, in main
    fix_code(sys.stdin.read(), args, encoding=encoding))
  File "/storage/lxdeng/.pyenv/versions/3.6.7/lib/python3.6/codecs.py", line 376, in write
    data, consumed = self.encode(object, self.errors)
UnicodeEncodeError: 'ascii' codec can't encode characters in position 78-79: ordinal not in range(128)
