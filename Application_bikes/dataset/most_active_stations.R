load("~/back_up_marco/CLPM/Application_bikes/dataset/wtab.RData")
Adj <- matrix(0,nrow = 780, ncol = 780)
for (idx in 1:nrow(wtab)){
  row = wtab[idx,]
  Adj[row[1,2], row[1,3]] = Adj[row[1,2], row[1,3]] + 1
  Adj[row[1,3], row[1,2]] = Adj[row[1,3], row[1,2]] + 1
}
deg <- rowSums(Adj)
names(deg) <- c(1:780)
sdeg <- sort(deg, decreasing = TRUE)
most_active <- as.numeric(names(sdeg[1:10]))

load("~/back_up_marco/CLPM/Application_bikes/dataset/stations_df.RData")
# most_active_stations <- stations_df[most_active,]
# save(most_active_stations, file = 'most_active_stations.RData')
#selection <- most_active_stations[1:3,] 

ids <- as.numeric(as.character(stations_df$id))
bool_pos <- ids %in% most_active
pos <- which(bool_pos == TRUE)
selection <- stations_df[pos,1:5]

##########
# Nodi bikes to track:
# Original ->     13,153,373 (C++ notation not R)
# Converted -> 0, 22, 50

### in Python
# sub_nodes, edgelist = get_sub_graph(edgelist_.copy(), 
#                                    type_of = type_of, 
#                                    n_sub_nodes = n_sub_nodes)
# edgelist, conversion = edgelist_conversion(edgelist,sub_nodes,n_nodes) 


