import tarfile

# Extract the cifar-10-python.tar.gz file
with tarfile.open('cifar-10-python.tar.gz', 'r:gz') as tar:
    tar.extractall()
    print("Extracted files:")
    for member in tar.getnames():
        print(member)