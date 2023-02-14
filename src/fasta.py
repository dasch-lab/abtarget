
def parse(path):

  with open(path, 'r') as fp:
    name, seq = None, []
    for line in fp:
      line = line.rstrip()
      if line.startswith(">"):

        # Append previous
        if name: 
            yield (name, ''.join(seq))

        name, seq = line[1:], []
      else:
        seq.append(line)

    # Append last in list
    if name: 
      yield (name, ''.join(seq))

def parse_handle(handle):

  name, seq = None, []
  for line in handle:
    line = line.rstrip()
    if line.startswith(">"):

      # Append previous
      if name: 
          yield (name, ''.join(seq))

      name, seq = line[1:], []
    else:
      seq.append(line)

  # Append last in list
  if name: 
    yield (name, ''.join(seq))