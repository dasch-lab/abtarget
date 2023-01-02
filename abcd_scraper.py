#!/usr/bin/env python

import os
import re
import argparse
import traceback
import sys

import urllib.request

from bs4 import BeautifulSoup
import pandas

def url_generator():

  alphabet = 'AA'
  suffix = 'ABCD_'
  counter = 1
  while True:
    value = '{}{}{:03d}'.format(suffix,alphabet,counter)
    if counter == 999:
      counter = 1

      if alphabet == 'ZZ':
        raise Exception('End of sequence')

      if alphabet[1] == 'Z':
        alphabet = '{0}A'.format(
          chr(ord(alphabet[0]) + 1)
        )
      else:
        alphabet = '{0}{1}'.format(
          alphabet[0],
          chr(ord(alphabet[1]) + 1),
        )
    else:
      # Increment counter
      counter += 1

    yield value

def scrape_url(url):

  try:

    # Fetch page
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request) as response:
      html = response.read()

    # No result
    if not html:
      raise Exception('Unable to fetch data')

    # Parse page
    soup = BeautifulSoup(html,features="html.parser")

    # Search for main container
    main = soup.find("div", {"class": "main-elmt"})
    if not main:
      raise Exception('Unable to find main element')

    # Extract data from table
    result = {}
    for row in main.findAll('tr'):
      header = row.find('th')
      data = row.find('td')
      if not header or not data:
        continue

      # Parse entry
      header = header.text.lower().replace(' ', '_')
      data = data.text
      result[header] = data

    return result


  except Exception as e:
    return None
  except urllib.error.URLError as e:
    print('URL {} does not exists'.format(url))
    return None

if __name__ == "__main__":

  # Initialize the argument parser
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-o', '--output', help='output file', dest='output', type=str, required=True)

  # Parse arguments
  args = argparser.parse_args()

  # Scrape each page from the database 
  counter = 0
  result = []
  for id in url_generator():

    # Generate the final url
    url = 'https://web.expasy.org/abcd/{}'.format(id)
    print('Fetching id {}'.format(id))

    # Scrape page
    data = scrape_url(url)
    if data is not None:

      # Append id & store
      data['id'] = id
      result.append(data)

    # Increment counter
    counter += 1
    if counter == 10:
      break

  # Store results
  result = pandas.DataFrame.from_dict(result)
  result.to_csv(args.output, sep='\t', index=False)

  # Done
  print('Scraped {} pages. Number of extracted entries: {}'.format(counter, len(result)))