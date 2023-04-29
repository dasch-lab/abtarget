## declare an array variable
declare -a arr=("1CIC_1.pdb" "1CIC_2.pdb" "1DVF_1.pdb" "1PG7_4.pdb" "1IAI_2.pdb" "3BQU_1.pdb" "4K7P_2.pdb" "4XHJ_1.pdb" "5FYK_2.pdb" "5N2K_8.pdb" "5JO4_1.pdb" "5JO4_2.pdb" "5XAJ_2.pdb" "5XAJ_3.pdb" "6HKG_2.pdb" "6JMQ_1.pdb" "6JMR_2.pdb") 
## now loop through the above array
for i in "${arr[@]}"
do
   
   IFS='
'
   for x in `grep -rnFi "REMARK 950 MOLECULE" "./dataset/abdb/NR_LH_Protein_Kabat/${i}"`; do 
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