### Note about data being saved to mongodb:
```
dbname:'kiddraw',
colname: 'tracing_eval'
iterationName: 'pilot0', 'pilot1', 'pilot2'
```

##### Stimuli Recording for Each IterationName
###### pilot0 
Dec.  2018
- Stimuli Database: kiddraw_eval_tracing
- Stimuli Types: square + star tracing, circle copying
- Participants: recruited about 1 participant per tracing image
###### pilot1
Jan. 6, 2019
- Stimuli Database: kiddraw_eval_tracing
- Stimuli Types: square + star tracing, circle copying
- Participants: recruited about 2 participant per tracing image
###### pilot2
Feb. 22, 2019
- Stimuli Database: kiddraw_eval_tracing_square_copy
- Stimuli Types: square copying
- Participants: recruited about 3 participant per tracing image
- what's new: 
    - a copying/tracing tag was added to the filename of all images and images are moved to https://s3.amazonaws.com/kiddraw-tracing-2.
    - migrate the old stimuli database to "kiddraw_eval_tracing" to "kiddraw_tracing_eval2". A new variable has_ref (True: tracing. False:copying) was added.
    - a new database "iddraw_eval_tracing_square_copy" was created for square copying images.

