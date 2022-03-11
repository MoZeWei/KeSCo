#include<string>
#include<vector>
#include<iostream>
#include<fstream>
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constant.h"
#include "llvm/Analysis/LoopInfo.h"
#include<unordered_map>
#include "Graph.h"

using namespace llvm;
using namespace graph;
//This pass is to insert timing function before & after each function and also analysis 
//data dependencies among functions.

//Construct graph for each function

namespace{

    struct Profiling : public ModulePass{
        static char ID;

        Profiling();
        ~Profiling();

        bool runOnModule(Module & M) override;
        std::string get_father_dir(std::string);
        void get_tool_library_func(Module & M);
        void getAnalysisUsage(AnalysisUsage &AU) const override;
        std::vector<Value*> GetKernelArgs(Node *, Module & M);
        void set_target_func(Module & M);                               //To locate kernel func
        void find_kernel_funcs(Module & M);
        BasicBlock * get_serial_last_bb(Function *);
        bool Find_TargetFunc(Function *);                               //Search target func by match name's prefix
        void End_Synchronize(Function *, Function *);
        //TO.DO.: Destroy streams

        //data
        std::string profiling_data_file_path;
        std::unordered_map<Function*,bool> tool_func_map;
        std::unordered_map<CallInst*,Node*> callinst_node_map;
        std::unordered_map<Function*,bool> target_func_map;
        std::unordered_map<Function*,bool> kernel_func_map;             //shows which function is kernel function
        std::vector<std::string> target_func_list;                      //We only have unmangling function name
    };

    Profiling::Profiling() : ModulePass(ID){
        target_func_list= {"FUNC"};
    }

    Profiling::~Profiling(){

    }

    void Profiling::getAnalysisUsage(AnalysisUsage &AU) const{
        AU.addRequired<LoopInfoWrapperPass>();
    }

    bool Profiling::runOnModule(Module & M){
        std::string father_path;
        std::string ModuleName = M.getModuleIdentifier();
        father_path = get_father_dir(ModuleName);
        std::string profiling_data_file_path = father_path+"profiling.txt";
        std::string data_dependency_graph_file_path = father_path+"DDG.txt";

        //This is to profile
        //TO.DO.: need an extra function to output the timing into file(TOOL LIBRARY)
        /*
        auto hipEvent_t = StructType::create(M.getContext(), "struct.ihipEvent_t");
        auto hipEvent_PtrT = PointerType::get(hipEvent_t,0);
        GlobalVariable * gpu_start_event_ptr = new GlobalVariable(M,hipEvent_PtrT,false,GlobalValue::CommonLinkage,
                                                                    0,"mzw_gpu_start");
        GlobalVariable * gpu_end_event_ptr = new GlobalVariable(M,hipEvent_PtrT,false,GlobalValue::CommonLinkage,
                                                                    0,"mzw_gpu_end");

        gpu_start_event_ptr->setAlignment(MaybeAlign(8));
        gpu_end_event_ptr->setAlignment(MaybeAlign(8));
        ConstantPointerNull * NULLPTR = ConstantPointerNull::get(hipEvent_PtrT);
        if(NULLPTR == nullptr)
        {
            errs()<<"cannot gen nullptr for hipevent\n";
            exit(1);
        }
        gpu_start_event_ptr->setInitializer(NULLPTR);
        gpu_end_event_ptr->setInitializer(NULLPTR);
        */

        //TO.DO.: Create global stream
        //1. define stream_type
        //2. announce global stream object
        //3. use hipcreateStream() to init stream
        auto cuStream_t = StructType::getTypeByName(M.getContext(), "struct.CUstream_st");
        auto cuStream_t_PtrT = PointerType::get(cuStream_t,0);
        auto cuStream_t_PtrPtrT = PointerType::get(cuStream_t_PtrT,0);
        ConstantPointerNull * NULLPTR = ConstantPointerNull::get(cuStream_t_PtrT);

        size_t stream_num = 10;
        GlobalVariable * stream_var_ptrs[stream_num];
        for(int i = 0; i < stream_num; i++)
        {
            stream_var_ptrs[i] = new GlobalVariable(M,cuStream_t_PtrT,false,GlobalValue::CommonLinkage,0,"mzw_s"+std::to_string(i));
            stream_var_ptrs[i]->setAlignment(MaybeAlign(8));
            
            stream_var_ptrs[i]->setInitializer(NULLPTR);
        }

        //TO.DO.: use hipcreateStream() to init stream in main()                DONE
        //define hipStreamCreate()
        std::vector<Type*> StreamCreate_Func_ArgT(1,cuStream_t_PtrPtrT);
        FunctionType * StreamCreate_i32_FuncType = FunctionType::get(Type::getInt32Ty(M.getContext()),StreamCreate_Func_ArgT,false);
        Function * CreateStreamFunc = nullptr;

        for(auto func = M.getFunctionList().begin(); func != M.getFunctionList().end(); func++)
        {
            Function * cur_func = dyn_cast<Function>(func);
            if(cur_func->getName().str() == "cudaStreamCreate")
            {
                CreateStreamFunc = cur_func;
            }
        }

        if(!CreateStreamFunc)
            CreateStreamFunc = Function::Create(StreamCreate_i32_FuncType,Function::ExternalLinkage,"cudaStreamCreate",M);
        
        //Init the stream
        for(auto func = M.getFunctionList().begin(), func_end = M.getFunctionList().end(); 
            func != func_end; func++)
        {
            Function * cur_func = dyn_cast<Function>(func);
            if(cur_func->getName().str() == "main")
            {
                // BasicBlock * first_bb = dyn_cast<BasicBlock>(cur_func->begin());
                // Instruction * first_inst = dyn_cast<Instruction>(first_bb->getFirstInsertionPt());

                //TO.DO.: Put the stream init after cudaSetDevice()
                //NOTE: There two ways: 
                //First: Invoke with conditional jmp  
                //          Normally, we will jmp to the basicblock of invoke_inst->getNormalDest()
                //Second: Directly Call
                //          Normally, just go to the next node of it
                bool flag = false;
                Instruction * insert_point_inst = nullptr;
                for(auto bb = cur_func->begin(), bb_end = cur_func->end(); bb != bb_end; bb++)
                {
                    if(flag) break;
                    for(auto inst = bb->begin(), inst_end = bb->end(); inst != inst_end; inst++)
                    {
                        if(isa<CallInst>(inst))
                        {
                            CallInst * call_inst = dyn_cast<CallInst>(inst);
                            Function * called_func = call_inst->getCalledFunction();
                            if(called_func != nullptr && called_func->getName().str() == "cudaSetDevice")
                            {
                                flag = true;
                                insert_point_inst = call_inst->getNextNode();
                                break;
                            }
                        }
                        if(isa<InvokeInst>(inst))
                        {
                            InvokeInst * invoke_inst = dyn_cast<InvokeInst>(inst);
                            Function * invoked_func = invoke_inst->getCalledFunction();
                            if(invoked_func != nullptr && invoked_func->getName().str() == "cudaSetDevice")
                            {
                                flag = true;
                                BasicBlock * normal_succ_bb = invoke_inst->getNormalDest();
                                insert_point_inst = dyn_cast<Instruction>(normal_succ_bb->getFirstInsertionPt());
                            }
                        }
                    }
                }

                if(insert_point_inst == nullptr)
                {
                    errs()<<"Cannot find cudaSetDevice\n";
                    BasicBlock * first_bb = dyn_cast<BasicBlock>(cur_func->begin());
                    insert_point_inst = dyn_cast<Instruction>(first_bb->getFirstInsertionPt());
                }
                IRBuilder<> builder(insert_point_inst);
                for(int i = 0; i < stream_num; i++)
                {
                    Value * cur_stream_ptr = dyn_cast<Value>(stream_var_ptrs[i]);
                    std::vector<Value*> args = {cur_stream_ptr};
                    if(!builder.CreateCall(StreamCreate_i32_FuncType,CreateStreamFunc,makeArrayRef(args)))
                    {
                        errs()<<"Cannot create call of cudaStreamCreate()\n";
                        exit(1);
                    }
                }
            }
        }


        //Get the ptr of DeviceSynchronize()
        std::vector<Type*> DeviceSync_Func_ArgT(0);
        FunctionType * DeviceSync_i32_FuncType = FunctionType::get(Type::getInt32Ty(M.getContext()),DeviceSync_Func_ArgT,false);
        Function * DeviceSync_Func = nullptr;
        for(auto func = M.getFunctionList().begin(), func_end = M.getFunctionList().end(); func != func_end; func++)
        {
            Function * cur_func = dyn_cast<Function>(func);
            if(cur_func->getName().str() == "cudaDeviceSynchronize")
            {
                DeviceSync_Func = cur_func;
            }
        }
        
        if(!DeviceSync_Func)
            DeviceSync_Func = Function::Create(DeviceSync_i32_FuncType,Function::ExternalLinkage,"cudaDeviceSynchronize",M);

        //In order to construct the ddg of each function(CPU function, GPU kernel and Memcpy)
        //First we need to identify them
        //For Kernel, we will use hipLaunchKernel to call kernel, so use it to recognize
        //For Memcpy, we will use hipMemcpy, so use it to recognize
        //For regular CPU function, we need user to add prefix of "cpu_" of each CPU function
        //Other functions in HIP and library functions will be treated as CPU functions.

        get_tool_library_func(M);
        set_target_func(M);
        find_kernel_funcs(M);
        errs()<<"Begin to loop\n";

        //In fact, we need to handle all branch and loop first, just make them as a sub-graph
        //then hash them into a set, in the second walk(handle the sequential instruction) ingnore
        //those hashed inst.
        //Also, in the first walk, should we identify input and output of each function?
        //TO.DO.:  Handle the Loop Graph
        /*
        for(Module::FunctionListType::iterator func = M.getFunctionList().begin(),
            func_end = M.getFunctionList().end(); func != func_end; func++)
        {
            Function * cur_func = dyn_cast<Function>(func);
            if(tool_func_map.find(cur_func) != tool_func_map.end()) continue;
            else if(cur_func->size() == 0) continue;

            LoopInfo & LI = getAnalysis<LoopInfoWrapperPass>(*cur_func).getLoopInfo();
            size_t loop_counter = 0;
            for(Loop * L : LI)        //this is for every outmost(top-level) loop?
            {
                errs()<<"Loop "<<++loop_counter<<"\n";
                //11-19 consider one bb, not nested loop
                for(BasicBlock * bb : L->getBlocks())
                {
                    for(BasicBlock::iterator inst = bb->begin(), inst_end = bb->end();
                        inst != inst_end; inst++)
                    {
                        
                    }
                }
            }
        }
        */

        //We use this map to collect graph of each function, and walk through main() to build the whole CFG
        std::unordered_map<Function *, Seq_Graph*> Func_SG_map;
        std::unordered_map<Function *, DAG*> Func_DAG_map;
        std::unordered_map<Function *, StreamGraph*> Func_StreamG_map;
        //NOTE: When we have a function, the first argument(int32) n means the following n arguments
        //are output values(also input values)

        //Walk to generate SQ
        for(Module::FunctionListType::iterator func = M.getFunctionList().begin(),
            func_end = M.getFunctionList().end(); func != func_end; func++)
        {
            Function * caller_func = dyn_cast<Function>(func);
            if(caller_func == nullptr)
            {
                errs()<<"Fail to get the caller func\n";
                exit(1);
            }
            if(caller_func->size() == 0)
            {
                //errs()<<"Func "<<caller_func->getName()<<" size is 0\n";
                continue;
            }
            
            if(tool_func_map.find(caller_func) != tool_func_map.end()) continue;
            else if(!Find_TargetFunc(caller_func)) continue;
            //TO.DO.: Make all target func begin with some prefix, so we can efficiently add more target func
            
            //We only care about the control flow in a function, we dont care the whole CFG
            //So we have a general garph in a function, but a sub-graph representing a loop / branch
            //is also a part of the general graph.
            
            Seq_Graph * SG;
            DAG * dag;
            StreamGraph * Stream_G;
            if(Func_SG_map.find(caller_func) != Func_SG_map.end() && 
                Func_DAG_map.find(caller_func) != Func_DAG_map.end() && Func_StreamG_map.find(caller_func) != Func_StreamG_map.end()) 
            {
                continue;
            }
            else
            {
                SG = new Seq_Graph();
                dag = new DAG();
                Stream_G = new StreamGraph(stream_num);
                Func_SG_map[caller_func] = SG;
                Func_DAG_map[caller_func] = dag;
                Func_StreamG_map[caller_func] = Stream_G;
            } 

            size_t inst_counter = 0;
            size_t kernel_counter = 0;
            //std::string func_name = func->getName().str();
            for(Function::iterator bb = func->begin(), bb_end = func->end(); bb != bb_end; bb++)
            {
                for(BasicBlock::iterator inst = bb->begin(), inst_end = bb->end(); inst != inst_end; inst++)
                {
                    inst_counter++;
                    if(isa<CallInst>(inst))
                    {
                        CallInst * call_inst = dyn_cast<CallInst>(inst);
                        Function * called_func = call_inst->getCalledFunction();
                        if(called_func == nullptr)
                        {
                            continue;
                        }
                        std::string called_func_name = called_func->getName().str();
                        if(kernel_func_map.find(called_func) != kernel_func_map.end())      //we meet a kernel_func callInst
                        {
                            //get the first argument as target kernel function
                            //errs()<<"We get one kernel call inst "<<*call_inst<<"\n";

                            Value * v = dyn_cast<Value>(call_inst->getArgOperand(0));
                            //errs()<<*v<<"\n";
                            Function * kernel_func = called_func;
                            //errs()<<kernel_func->getName();
                            //errs()<<"\n";

                            //If it's in the kernel function IR, it will call itself(only with the same name of kernel function) again
                            //So we need CallInst as identifier.
                            //QUES.: Should we do more about this?
                            GPUFuncNode * node = new GPUFuncNode(call_inst,kernel_func,true,true);
                            
                            //node->CollectBBs(M);
                            //errs()<<"Got all BBs\n";
                            //Selece Stream for kernel node (Only For DEBUG)
                            //node->SetStream(stream_var_ptrs[kernel_counter % stream_num]);

                            //node->dump_inst();
                            //node->dumpBBs();
                            //errs()<<"***************\n";
                            
                            Node * prev_node = SG->get_last_Node();
                            SG->Insert(node,prev_node);

                            std::vector<Value*> input_args = GetKernelArgs(node,M);             //Here what we got is the original value of each arguments(Only Pointer)
                            //errs()<<"Got kernel args\n";

                            if(input_args.size() == 0)
                            {
                                std::cout<<"Cannot get input arguments of kernel "<<kernel_func->getName().str()<<std::endl;
                                exit(1);
                            }

                            //Register Input/Output Value into node
                            Value * output_value_n = input_args[0];
                            if(ConstantInt * CI = dyn_cast<ConstantInt>(output_value_n))
                            {
                                size_t output_n = CI->getZExtValue();
                                size_t n = kernel_func->arg_size();
                                errs()<<kernel_func->getName()<<" has "<<n<<" input arguments and "<<output_n<<" output arguments\n";
                                //We dont get the counter argument
                                //QUES.: Would it be good to record the counter argument?
                                for(size_t i = 1; i < 1+output_n; i++)
                                {
                                    node->add_output_value(input_args[i]);
                                }
                                for(size_t i = 1+output_n; i < n; i++)
                                {
                                    //We record the output_counter, but we need to ignore it when construct input_val in DAG
                                    node->add_input_value(input_args[i]);
                                }
                            }
                            else
                            {
                                errs()<<"1st argument of Function "<<called_func_name<<" is not constant value\n";
                                exit(1);
                            }
                            
                            kernel_counter++;

                        }
                        else if(called_func_name.find("cpu_") != std::string::npos)
                        {
                            errs()<<"CPU FUNCTION NOT SUPPORTED\n";
                            exit(1);
                            //This is a cpu function
                            Function * cpu_func = dyn_cast<Function>(call_inst->getFunction());
                            //TO.DO.: Hard to grep the true original input value
                            CPUFuncNode * node = new CPUFuncNode(call_inst,cpu_func,false,true);
                            Node * prev_node = SG->get_last_Node();
                            SG->Insert(node,prev_node);
                        }
                        else if(called_func_name == "cudaMemcpy")
                        {
                            //This is memcpy
                            Function * memcpy_func = dyn_cast<Function>(call_inst->getCalledFunction());
                            MemcpyNode * node = new MemcpyNode(call_inst,memcpy_func,false,false);             //QUES.: Should the gpu_flag be true?
                            //node->CollectBBs(M);

                            //node->dump_inst();
                            //node->dumpBBs();
                            //errs()<<"***************\n";
                            
                            Node * prev_node = SG->get_last_Node();
                            SG->Insert(node,prev_node);
                        }
                        else if(called_func_name == "cudaMemPrefetchAsync")
                        {
                            Function * prefetch_func = dyn_cast<Function>(call_inst->getCalledFunction());
                            PrefetchNode * node = new PrefetchNode(call_inst, prefetch_func, false, false);
                            Node * prev_node = SG->get_last_Node();
                            SG->Insert(node,prev_node);
                        }
                        else
                        {
                            //These are considered to be cpu function(how about hipblasGemmEx etc.?)
                            //In 12-4, we only consider about all Kernel launch without other instructions
                            //TO.DO.: But hipMalloc should be specified to handle.
                            //Function * cpu_library_func = dyn_cast<Function>(call_inst->getCalledFunction());
                            //errs()<<"we meet "<<cpu_library_func->getName()<<"\n";
                            //CPUFuncNode * node = new CPUFuncNode(call_inst,cpu_library_func,false,false);
                            //Node * prev_node = SG->get_last_Node();
                            //SG->Insert(node,prev_node);
                        }
                    }
                    else
                    {
                        //These instructions will be collected by data-flow analysis of each call_inst
                        //For those instructions might not be collected after all these, maybe it's time to kick off
                        //collect them as InstNode and bundle related ones into one SuperNode
                        //So now we dont have to handle them here.
                    }
                }
            }

            

            //TO.DO.: Print out the function Seq-Graph                              DONE
            errs()<<"\n\nThis is Seq_Graph for Function "<<caller_func->getName()<<"\n";
            SG->CollectBBs(M);
            SG->Print_allNode();

            //Construct DAG
            //dag->dump();
            //errs()<<"Begin to construct DAG from SG\n";
            dag->ConstructFromSeq_Graph(SG);
            dag->dump();
            dag->levelize();
            //dag->dump_level();
            //TO.DO.: Dump the linkage among nodes

            //Distribute stream to kernel and add right event (Not moving bb yet)
            dag->StreamDistribute(Stream_G,stream_var_ptrs,M);
            Stream_G->dump_Graph();
            //TO.DO.: Delete duplicated/useless events
            Stream_G->fix_EEs();
            Stream_G->dump_EEs();

            
            //errs()<<"Begin to change order of func call\n";
            BasicBlock * first_bb = dyn_cast<BasicBlock>(func->getBasicBlockList().begin());
            BasicBlock * tail_bb = get_serial_last_bb(caller_func);
            Node * EntryNode = dag->get_EntryNode();
            Stream_G->FuncCall_Reorder(first_bb,tail_bb,EntryNode);                 //QUES.: Is this really helpful?
            


            errs()<<"Begin to insert Event\n";
            Stream_G->create_Events(M,first_bb,stream_var_ptrs);

            errs()<<"Begin to sync\n";
            End_Synchronize(caller_func,DeviceSync_Func);


            //TO.DO.: Dump BBs of each node to check
        }



        return true;
    }

    BasicBlock * Profiling::get_serial_last_bb(Function * func)
    {
        BasicBlock * res;
        for(auto cur_bb = func->begin(), bb_end = func->end(); cur_bb != bb_end; cur_bb++)
        {
            res = dyn_cast<BasicBlock>(cur_bb);
        }
        //errs()<<"Last BB is ";
        //res->printAsOperand(errs(),false);
        //errs()<<"\n";
        return res;
    }


    std::string Profiling::get_father_dir(std::string file_path)
    {
        std::string res = file_path;
        int n = file_path.length();
        int i = n - 1;
        for(; i >= 0; i--)
        {
            if(file_path[i] == '/') break;
        }
        int file_name_length = n - 1 - i;
        while(file_name_length >0)
        {
            file_name_length--;
            res.pop_back();
        }
        //std::cout<<"Father path is "<<res<<std::endl;
        return res;
    }

    void Profiling::get_tool_library_func(Module & M)
    {
        //TO.DO.: Profiling helper func
        
    }

    void Profiling::set_target_func(Module & M)
    {
        for(auto func = M.getFunctionList().begin(), func_end = M.getFunctionList().end();
            func != func_end; func++)
        {
            Function * cur_func = dyn_cast<Function>(func);
            std::string func_name = cur_func->getName().str();
            for(auto target_func_name : target_func_list)
            {
                if(func_name.find(target_func_name) != std::string::npos)
                {
                    target_func_map[cur_func] = true;
                }
            }
        }
    }

    std::vector<Value *> Profiling::GetKernelArgs(Node * node, Module & M)
    {
        CallInst * call_inst = node->getCallInst();
        //Function * called_func = node->getFunction();

        //errs()<<called_func->getName().str()<<":\n";

        std::vector<Value *> res;
        
        Value * output_counter_value = call_inst->getArgOperand(0);
        size_t Args_num = call_inst->getNumArgOperands();
        
        res.push_back(output_counter_value);
        
        for(size_t i = 1; i < Args_num; i++)
        {
            Value * cur_argv = call_inst->getArgOperand(i);
            Type * v_type = cur_argv->getType();
            if(isa<PointerType>(v_type))
            {
                LoadInst * ld_inst = nullptr;
                if(isa<LoadInst>(cur_argv))
                    ld_inst = dyn_cast<LoadInst>(cur_argv);
                else if(isa<AllocaInst>(cur_argv))
                {
                    //it may be passed value by llvm.memcpy
                    //For now, just ignore
                    res.push_back(cur_argv);
                    continue;
                }
                else
                {
                    if(BitCastInst * bc_inst = dyn_cast<BitCastInst>(cur_argv))
                    {
                        //errs()<<"found bc_inst for "<<cur_argv<<"\n";
                        Value * tmp_v = bc_inst->getOperand(0);
                        ld_inst = dyn_cast<LoadInst>(tmp_v);
                    }
                    else
                    {
                        errs()<<"Wrong bc_inst\n";
                        exit(1);
                    }
                }

                Value * cur_argv_addr = ld_inst->getOperand(0);
                for(auto user = cur_argv_addr->user_begin(), user_end = cur_argv_addr->user_end();
                    user != user_end; user++)
                {
                    //the argv_addr is only stored once
                    if(StoreInst * st_inst = dyn_cast<StoreInst>(*user))
                    {
                        Value * original_v = st_inst->getOperand(0);
                        res.push_back(original_v);
                    }
                }
            }
            else
            {
                res.push_back(cur_argv);
            }
        }
        errs()<<"The input operand of "<<*call_inst<<" are:\n";
        for(auto arg : res) errs()<<*arg<<"\n";
        return res;
    }

    void Profiling::find_kernel_funcs(Module & M)
    {
        for(auto func = M.getFunctionList().begin(), func_end = M.getFunctionList().end(); 
            func != func_end; func++)
        {
            Function * caller_func = dyn_cast<Function>(func);
            for(auto bb = func->begin(), bb_end = func->end(); bb != bb_end; bb++)
            {
                for(auto inst = bb->begin(), inst_end = bb->end(); inst != inst_end; inst++)
                {
                    if(isa<CallInst>(inst))
                    {
                        CallInst * callinst = dyn_cast<CallInst>(inst);
                        Function * called_func = callinst->getCalledFunction();
                        if(called_func != nullptr && called_func->getName().str() == "cudaLaunchKernel")
                        {
                            kernel_func_map[caller_func] = true;
                        }
                    }
                }
            }
        }   
    }

    bool Profiling::Find_TargetFunc(Function * func)
    {
        std::string target_func_name = func->getName().str();
        for(std::string candidate_func_name : target_func_list)
        {
            if(target_func_name.find(candidate_func_name.c_str()) != std::string::npos)
            {
                errs()<<"Matched target func: "<<target_func_name<<"\n";
                errs()<<"Matched candidate func: "<<candidate_func_name<<"\n";
                return 1;
            }
        }
        return 0;
    }

    void Profiling::End_Synchronize(Function * func, Function * DeviceSync_Func)
    {
        for(auto bb = func->begin(), bb_end = func->end(); bb != bb_end; bb++)
        {
            for(auto inst = bb->begin(), inst_end = bb->end(); inst != inst_end; inst++)
            {
                Instruction * cur_inst = dyn_cast<Instruction>(inst);
                if(isa<ReturnInst>(cur_inst))
                {
                    IRBuilder<> builder(cur_inst);
                    std::vector<Value*> args = {};
                    Value * ret_v = builder.CreateCall(DeviceSync_Func,makeArrayRef(args));
                    if(!ret_v)
                    {
                        errs()<<"Cannot create DeviceSynchronize func call inst\n";
                        exit(1);
                    }
                }
            }
        }
    }

}

char Profiling::ID = 0;

static RegisterPass<Profiling> X("profiling","A Pass to Profile and run data analysis",
                                true, false);

static RegisterStandardPasses Y(
    PassManagerBuilder::EP_EarlyAsPossible,
    [] (const PassManagerBuilder & Builder, 
    legacy::PassManagerBase & PM) {PM.add(new Profiling());});
