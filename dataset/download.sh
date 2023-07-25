#for user in "$@" 
#do
#    echo "$user";
#done

# Get CIFAR10
#wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
#tar -xzvf cifar-10-python.tar.gz
#rm cifar-10-python.tar.gz

# Get CIFAR100

# Get CINIC10
mkdir cinic-10 && cd cinic-10
wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz
tar -xzvf CINIC-10.tar.gz
rm CINIC-10.tar.gz
cd ../

