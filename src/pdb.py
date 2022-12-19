import re
import sys
import struct

'''
A table to convert three-letters code AAs to one letter
'''
aaTable = {
  'ALA':'A',
  'ARG':'R',
  'ASN':'N',
  'ASP':'D',
  'ASX':'B',
  'CYS':'C',
  'GLU':'E',
  'GLN':'Q',
  'GLX':'Z',
  'GLY':'G',
  'HIS':'H',
  'ILE':'I',
  'LEU':'L',
  'LYS':'K',
  'MET':'M',
  'PHE':'F',
  'PRO':'P',
  'SER':'S',
  'THR':'T',
  'TRP':'W',
  'TYR':'Y',
  'VAL':'V'
}

import re
import struct

'''
A table to convert three-letters code AAs to one letter
'''
aaTable = {
  'ALA':'A',
  'ARG':'R',
  'ASN':'N',
  'ASP':'D',
  'ASX':'B',
  'CYS':'C',
  'GLU':'E',
  'GLN':'Q',
  'GLX':'Z',
  'GLY':'G',
  'HIS':'H',
  'ILE':'I',
  'LEU':'L',
  'LYS':'K',
  'MET':'M',
  'PHE':'F',
  'PRO':'P',
  'SER':'S',
  'THR':'T',
  'TRP':'W',
  'TYR':'Y',
  'VAL':'V'
}
def _aminoacid(code):

  if code in aaTable:
    return aaTable[code]

  return 'X'

def info(path, chain=None):

  regexList = {
    'molecule': '^(?:REMARK\s+\d+\s+)(\w+)(?:\s+\:\s+)(\w)(?:\s+\:\s+)([^\n]+)$',
    'chain': '^(?:REMARK\s+\d+\s+)(?:CHAIN)\s+(\w)\s+(\w)\s+(\w)$'
  }

  chainLut = {}
  resultMap = {}
  # if chain:
  #   resultMap[chain] = {
  #     'organism': 'unknown',
  #     'description': 'unknown'
  #   }
  remark_re = re.compile('^(?:REMARK\s+\d+\s+)(\w+)(?:\s+\:\s+)(\w)(?:\s+\:\s+)([^\n]+)$')
  with open(path) as handle:
    for line in handle:

      line = line.strip()
      if not line.startswith('REMARK'):
        continue

      for key in regexList:
        match = re.search(regexList[key], line)
        if not match:
          continue

        if key == 'molecule' and match.group(1) == 'MOLECULE':
          remark_chain = match.group(2)
          remark_var = match.group(3)
          if remark_chain not in resultMap:
            resultMap[remark_chain] = {}

          resultMap[remark_chain]['description'] = remark_var
          break

        if key == 'molecule' and match.group(1) == 'SPECIES':
          remark_chain = match.group(2)
          remark_var = match.group(3)
          if remark_chain not in resultMap:
            resultMap[remark_chain] = {}

          resultMap[remark_chain]['organism'] = remark_var
          break

        if key == 'chain':
          remark_chain = match.group(1)
          remark_true = match.group(3)
          chainLut[remark_chain] = remark_true
          break

        raise Exception('Invalid remark: {0}'.format(line))

  # if chain not in chainLut:
  #   return {
  #     'organism': 'unknown',
  #     'description': 'unknown' 
  #   }
  result = {
    'organism': 'unknown',
    'description': 'unknown'
  }
  for key in ['H', 'L']:
    if key not in chainLut:
      continue

    # print(resultMap)
    key_index = chainLut[key]
    if key_index not in resultMap:
      continue

    if 'organism' in resultMap[key_index] and result['organism'] == 'unknown':
      result['organism'] = resultMap[key_index]['organism']

    if 'description' in resultMap[key_index] and result['description'] == 'unknown':
      result['description'] = resultMap[key_index]['description']

  if chain not in chainLut:
    return None
    
  chain = chainLut[chain]
  if chain not in resultMap:
    return None

  result.update(resultMap[chain])
  return result
  # resultMap[chain].update(default)
  # print(resultMap[chain])
  
  # return resultMap[chain]
     
def sequence(path, chain, gapped):

  '''
  Extract the sequence from the pdb
  '''

  sequence = ''
  status = {
    'chain': None,
    'resnum': 0
  }
  for line in parse(path):

    chain = line['chain'] if not chain else chain
    if line['chain'] != chain:
      continue

    if line['resnum'] < status['resnum']:
      break

    # Append gaps
    if gapped and status['resnum'] != line['resnum']:
      while status['resnum'] < line['resnum']-1:
        sequence += '-'
        status['resnum'] += 1

    # Update sequence
    if status['resnum'] != line['resnum']:
      sequence += _aminoacid(line['resname'])

    # Update status
    status['chain'] = line['chain']
    status['resnum'] = int(line['resnum'])

  return sequence

# def parse(path):

#   '''
#   Parse the PDB line by line and return the parsed atom line
#   '''

#   fieldWidths = (
#     6,  # record
#     5,  # atom serial
#     -1, 
#     4,  # atom name
#     -1, 
#     3, # res name
#     -1,
#     1, # chain id
#     4, # res number
#     1  # AChar
#   )

#   formatString = ' '.join('{}{}'.format(abs(fw), 'x' if fw < 0 else 's') for fw in fieldWidths)
#   fieldstruct = struct.Struct(formatString)
#   unpack = fieldstruct.unpack_from
#   atomParse = lambda line: tuple(s.decode().strip() for s in unpack(line.encode()))

#   # Iterate the pdb file
#   coord_re = re.compile('^(ATOM)')
#   with open(path) as handle:
#     for line in handle:

#       # Skip everything but coordinate lines
#       if not coord_re.match(line):
#         continue

#       # Parse coordinate line
#       data = atomParse(line)
#       yield {
#         'record': data[0],
#         'serial': data[1],
#         'name': data[2],
#         'resname': data[3],
#         'chain': data[4],
#         'resnum': int(data[5]),
#         'achar': data[6],
#         'raw': line
#       }


# def sequence(path, chain=None):

#   '''
#   Extract the sequence from the pdb
#   '''

#   sequence = ''
#   status = {
#     'chain': None,
#     'resnum': None
#   }
#   for line in parse(path):

#     chain = line['chain'] if not chain else chain
#     if line['chain'] != chain:
#       continue

#     # Update sequence
#     if status['resnum'] != line['resnum']:
#       sequence += _aminoacid(line['resname'])

#     # Update status
#     status['chain'] = line['chain']
#     status['resnum'] = line['resnum']

#   return sequence

def parse(path):

  '''
  Parse the PDB line by line and return the parsed atom line
  '''

  fieldWidths = (
    6,  # record
    5,  # atom serial
    -1, 
    4,  # atom name
    -1, 
    3, # res name
    -1,
    1, # chain id
    4, # res number
    1  # AChar
  )

  formatString = ' '.join('{}{}'.format(abs(fw), 'x' if fw < 0 else 's') for fw in fieldWidths)
  fieldstruct = struct.Struct(formatString)
  unpack = fieldstruct.unpack_from
  atomParse = lambda line: tuple(s.decode().strip() for s in unpack(line.encode()))

  # Iterate the pdb file
  coord_re = re.compile('^(ATOM)')
  with open(path) as handle:
    for line in handle:

      # Skip everything but coordinate lines
      if not coord_re.match(line):
        continue

      # Fix issues with columns
      # print(line[12])
      # if line[12] 
      # print(line[12])
      if line[11] != ' ':
        atom = line[0:12]
        # print(atom + '|')
        atom = atom.replace('ATOM', '').strip()
        atom = 'ATOM' + atom.rjust(7, ' ')
        # atom = atom + ' ' if atom[10] == ' ' else ''
        # print(atom + '|')
        # print(line)
        line = atom + line[12:]
        # print(line)
      # sys.exit(1)

      # Parse coordinate line
      data = atomParse(line)
      # print(data)
      yield {
        'record': data[0],
        'serial': data[1],
        'name': data[2],
        'resname': data[3],
        'chain': data[4],
        'resnum': int(data[5]),
        'achar': data[6],
        'raw': line
      }