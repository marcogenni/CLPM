import sys
sys.path.append('../')
from CLPM_functions import *

np.random.seed(12345)
torch.manual_seed(54321)

# epochs = 2000
period = 1.
frames_btw = 60
# penalty_d = 20
# penalty_p = 250
snapshot_times = ['2009/06/29 09:21:19', 
                  '2009/06/29 10:06:15',
                  '2009/06/29 12:05:23',
                  '2009/06/29 14:07:23',
                  '2009/06/29 15:47:58',
                  '2009/06/29 16:37:54',
                  '2009/06/29 18:49:10',
                  '2009/06/29 20:44:01'
                   ]

fnames = ['app_acm_dist_' + str(idx) + '.pdf' for idx in range(1,9)]                  


ClpmPlot(model_type = 'distance',
         dpi = 250,
         period = period,
         size = (1200,900),
         is_color = True,
         formato = 'mp4v',
         frames_btw = frames_btw,
         nodes_to_track = [None],
         sub_graph = True,
         type_of = 'degree',
         n_hubs = 2,
         n_sub_nodes = 60,
         start_date = (2009,6,29,8,0),
         end_date = (2009,6,29,20,59)
         )

# ClpmSnap(extraction_times = snapshot_times,
#          filenames=fnames,
#          model_type = 'distance',
#          dpi = 250,
#          period = period,
#          size = (1200,900),
#          is_color = True,
#          formato = 'mp4v',
#          frames_btw = frames_btw,
#          nodes_to_track = [None],
#          sub_graph = True,
#          type_of = 'degree',
#          n_hubs = 2,
#          n_sub_nodes = 60,
#          start_date = (2009,6,29,8,0),
#          end_date = (2009,6,29,20,59)
#          )

