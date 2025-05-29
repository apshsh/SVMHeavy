import sys

def testfn(y,z):
  return((2*float(y)*float(y))+((float(z)-1)*(float(z)-1)))

if __name__ == "__main__":
  y = sys.argv[2]
  z = sys.argv[4]
  with open("pyres.txt", "w") as f:
    print("{ null 0 [",testfn(y,z),"] }", file=f)
  #print(testfn(y,z))
