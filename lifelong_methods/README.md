# Lifelong Learning Methods Guide
The package documentation is [here](https://iirc.readthedocs.io/en/latest/lifelong_methods.html)

The lifelong learning methods in this package follow the following procedures

```
example_model = lifelong_methods.methods.example.Model(args)  # replace example with whatever module is there

for task in tasks:
    task_data <- load here the task data
    
    # This method initializes anything that needs to be inizialized at the beginning of each task
    example_model.prepare_model_for_new_task(task_data, **kwargs) 

    for epoch in epochs:
        # Training
        for minibatch in task_data:
            # This is where the training happens
            predictions, loss = example_model.observe(minibatch)

        # This is where anything that needs to be done after each epoch should be done, if any
        example_model.consolidate_epoch_knowledge(**kwargs) 
    
    # This is where anything that needs to be done after the task is done takes place
    example_model.consolidate_task_knowledge(**kwargs)

    # Inference
    # This is where the inference happens
    predictions = example_model(inference_data_batch)
```

This is the typical order of how things flow in a lifelong learning scenario, and how does this package handles 
that. This order makes it easy to implement new methods with shared base, so that they can run using the same code and 
experimenting can be fast

When defining a new lifelong learning model, the first step is to create a model that inherits from 
*lifelong_methods.methods.base_method.BaseMethod*, then the following abstract methods need to be defined (see the methods 
docs for more details), private methods here are run from inside their similar but public methods so that shared stuff 
between the different methods doesn't need to be reimplemented (like resetting the scheduler after each task, etc), 
see the docs to know what is already implemented in the public methods so that you don't reimplement them:
*  *_prepare_model_for_new_task*: This private method is run from inside the *prepare_model_for_new_task* in the 
*BaseMethod*, 
*  *_consolidate_epoch_knowledge*: This private method is run from inside the *consolidate_epoch_knowledge* in the 
*BaseMethod*   
*  *observe*
* *forward*
* *consolidate_task_knowledge*
