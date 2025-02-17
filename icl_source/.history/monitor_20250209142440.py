我用的算时间的方法：
else:
                    # eval_result = trainer.evaluate(eval_dataset=test_dataset)
                    # eval_result = trainer.evaluate(eval_dataset=test_dataset)
                    small_test_dataset = Subset(test_dataset, range(10))  # 取前10条数据
                    dataloader = DataLoader(small_test_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])  # 取出单个样本

                    times = []  # 存储每个样本的推理时间
                    peak_memories = []
                    for idx, sample in enumerate(dataloader):
                        torch.cuda.reset_peak_memory_stats() 
                        with torch.no_grad():  # 确保不计算梯度
                            start_time = time.time()
                            eval_result = trainer.predict([sample])  # 这里使用 `predict`
                            end_time = time.time()

                        inference_time = end_time - start_time
                        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
                        times.append(inference_time)
                        peak_memories.append(peak_memory)
                        print(f"Sample {idx}: Inference time = {inference_time:.4f} sec, Peak memory = {peak_memory:.4f} GB")

                    mean_time = np.mean(times)
                    std_time = np.std(times)
                    mean_mem = np.mean(peak_memories)
                    std_mem = np.std(peak_memories)

                    print(f"Mean inference time: {mean_time:.4f} sec, Std: {std_time:.4f} sec")
                    print(f"Mean peak memory: {mean_mem:.4f} GB, Std: {std_mem:.4f} GB")
                    eval_result = trainer.evaluate(eval_dataset = small_test_dataset)