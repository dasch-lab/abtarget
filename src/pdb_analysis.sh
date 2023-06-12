## declare an array variable
declare -a arr=("6dc8.pdb" "4rgo.pdb" "6a0z.pdb" "6i3z.pdb" "5esz.pdb" "6pxr.pdb" "8dcy.pdb" "7vng.pdb") 
## now loop through the above array
for i in "${arr[@]}"
do
   
   IFS='
'
   for x in `grep -rnFi "REMARK 950 MOLECULE" "./disk1/abtarget/dataset/sabdab/protein/${i}"`; do 
    char=${x:25:1}
    #echo $char;
    #if (( "$char" != "H"  && "$char" != "L" )) 
    #if [ "$char" -ne "H" ] && [ "$char" -ne "L" ]; then
    if [ "$char" != "H" ] && [ "$char" != "L" ]; then
        echo "$i $char ${x:28:200}";
    fi
   done;
  # 
   # or do whatever with individual element of the array
done