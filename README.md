Standard implementation is SHA3-256 f[1600] with 24 rounds, can be changed in code

Provide path(s) of files to encrypt as arguments to the launching of the code

Parameters:
- LOGG: logging your timings to a file (default True)
- total: total State size (default 1600)
- rate: rate size of State (default 1088 aka 136B)
- delim_begin, delim_end : padding settings (default 0x06 and 0x80 respectively)
