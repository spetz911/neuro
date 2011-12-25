#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

def main():
	if len(sys.argv) == 1:
		i = sys.stdin
		f = sys.stdout
		o = [x.replace('>> ', '') for x in i.readlines()]
	elif len(sys.argv) == 2:
		i = sys.stdin
		f = open(sys.argv[1],'w')
		o = ["<BR>" + x.replace('>> ', '') for x in i.readlines()]
	elif len(sys.argv) == 3:
		i = open(sys.argv[1])
		f = open(sys.argv[2],'w')
		o = ["<BR>" + x.replace('>> ', '') for x in i.readlines()]
	[f.write(x) for x in o]
	f.close()

if __name__ == '__main__':
	try:
		main()
	except ValueError:
		print("error" + str(ValueError))
