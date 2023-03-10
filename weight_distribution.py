print(weights)
j_min = min_weight
j_max = max_weight
for i in list(weights):
    if i==min_weight:
        j_min+=1
    elif i==max_weight:
        j_max+=1
print(str(j_min)+" out of "+str(len(weights))+" are minimum, i.e. w = "+str(min_weight))
print("Minimum Weight is "+str(min(weights)))
print(str(j_max)+" out of "+str(len(weights))+" are maximum, i.e. w = "+str(max_weight))
print("Maximum Weight is "+str(max(weights)))
plt.figure()
ax = sns.distplot(weights,axlabel="Distribution of Final Weights")
plt.show()