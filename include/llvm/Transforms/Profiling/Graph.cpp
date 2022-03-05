#include<string>
#include<vector>
#include<iostream>
#include<fstream>
#include<queue>
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
#include<unordered_map>
#include "Graph.h"
#include "llvm/IR/CFG.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"


using namespace llvm;

namespace graph{

    Node::Node()
    {
        undone_pred = 0; 
        succ.clear(); 
        pred.clear();
    }

    Node::~Node()
    {
        //do nothing
    }

    void Node::clear_succ(Node * target = nullptr)
    {
        if(target == nullptr)   succ.clear();
        else
        {
            size_t i = 0;
            for(; i < succ.size(); i++)
            {
                if(target == succ[i]) break;
            }
            if(i >= succ.size())
            {
                std::cout<<"Cannot clear target succ node"<<std::endl;
                exit(1);
            }
            else
            {
                succ.erase(succ.begin() + i);
            }
        }
    }

    void Node::clear_pred(Node * target = nullptr)
    {
        if(target == nullptr)
        {
            pred.clear();
            undone_pred = 0;
        }
        else
        {
            size_t i = 0;
            for(; i < pred.size(); i++)
            {
                if(target == pred[i]) break;
            }
            if(i >= pred.size())
            {
                std::cout<<"Cannot clear target pred node"<<std::endl;
                exit(1);
            }
            else
            {
                pred.erase(pred.begin() + i);
            }
        }
    }

    void Node::add_succ(Node * node)
    {
        succ.push_back(node);
    }

    void Node::add_pred(Node * node)
    {
        pred.push_back(node);
        undone_pred++;
    }

    size_t Node::get_pred_num()
    {
        return pred.size();
    }

    size_t Node::get_succ_num()
    {
        return succ.size();
    }

    Node * Node::get_pred(size_t i)
    {
        if(i >= pred.size()) return nullptr;
        else return pred[i];
    }

    Node * Node::get_succ(size_t i)
    {
        if(i >= succ.size()) return nullptr;
        else return succ[i];
    }

    void Node::dump_func()
    {

    }

    void Node::dump_inst()
    {

    }

    void Node::add_input_value(Value * v)
    {
        input_value.push_back(v);
    }

    void Node::add_output_value(Value * v)
    {
        output_value.push_back(v);
    }

    std::vector<Value*> Node::get_input_value()
    {
        return input_value;
    }

    std::vector<Value*> Node::get_output_value()
    {
        return output_value;
    }

    void Node::setCalledFunc(Function * f)
    {

    }

    void Node::setCallInst(CallInst * ci)
    {

    }

    CallInst * Node::getCallInst()
    {
        return nullptr;
    }

    Function * Node::getFunction()
    {
        return nullptr;
    }

    void Node::addBB(BasicBlock * bb)
    {
        bbs.push_back(bb);
    }

    void Node::deleteBB(BasicBlock * bb)
    {
        size_t i = 0;
        for(; i < bbs.size(); i++)
        {
            if(bb == bbs[i])
                break;
            else continue;
        }
        if(i == bbs.size())
        {
            std::cout<<"Cannot delete bb\n";
        }
        else 
        {
            bbs.erase(bbs.begin() + i);
        }
    }

    void Node::dumpBBs()
    {
        for(BasicBlock * bb : bbs)
        {
            bb->printAsOperand(errs(),false);
            errs()<<" ";
        }
        errs()<<"\n";
    }

    void Node::CollectBBs(Module & M)
    {
        //do nothing
    }

    std::vector<BasicBlock*> Node::getBBs()
    {
        std::vector<BasicBlock*> res;
        size_t num = bbs.size();
        for(size_t i = 0; i < num; i++)
        {
            res.push_back(bbs[i]);
        }
        return res;
    }

    void Node::SetStream(GlobalVariable * gv, Module & M)
    {
        //do nothing
    }

    InstNode::InstNode(Instruction * target_inst)
    {
        inst = target_inst;
    }

    InstNode::~InstNode()
    {
        //do nothing.
    }

    FuncNode::FuncNode(CallInst * ci, Function * cf, bool gpu_f, bool graph_f)
    {
        call_inst = ci;
        called_func = cf;
        gpu_flag = gpu_f;
        has_graph = graph_f;
    }

    FuncNode::~FuncNode()
    {
        //nothing to do
    }

    std::string FuncNode::get_func_name()
    {
        return called_func->getName().str();
    }

    void FuncNode::dump_func()
    {
        std::cout<<get_func_name()<<std::endl;
    }

    void FuncNode::dump_inst()
    {
        errs()<<*call_inst<<"\n";
    }

    void FuncNode::setCallInst(CallInst * ci)
    {
        call_inst = ci;
    }

    void FuncNode::setCalledFunc(Function * F)
    {
        called_func = F;
    }

    CallInst * FuncNode::getCallInst()
    {
        return call_inst;
    }

    Function * FuncNode::getFunction()
    {
        return called_func;
    }

    void FuncNode::CollectBBs(Module & M)
    {
        //do nothing
    }

    void FuncNode::SetStream(GlobalVariable * stream, Module & M)
    {
        //do nothing
    }

    GPUFuncNode::GPUFuncNode(CallInst * ci, Function * cf, bool gpu_f, bool graph_f):FuncNode(ci,cf,gpu_f,graph_f)
    {
        //do nothing?
    }

    void GPUFuncNode::CollectBBs(Module & M)
    {
        BasicBlock * cur_bb = this->getCallInst()->getParent();
        //Normally, in sequential kernel launches, one bb containing a kernel only has one predecessor
        //QUES.: How about if-else or loop?
        //dump_inst();
        //errs()<<"Its bb has "<<pred_size(cur_bb)<<" pred bbs\n";
        std::vector<BasicBlock*> bbs_to_split;
        std::vector<Instruction*> splited_node_insts;

        for(BasicBlock * direct_pred_bb : predecessors(cur_bb))
        {
            bool one_bb_split_flag = false;
            for(auto inst = direct_pred_bb->begin(), inst_end = direct_pred_bb->end();
                inst != inst_end; inst++)
            {
                if(one_bb_split_flag) break;
                if(isa<CallInst>(inst))
                {
                    CallInst * Dim3_Inst = dyn_cast<CallInst>(inst);
                    if(Dim3_Inst->getCalledFunction()->getName().str().find("dim3") != std::string::npos)      //There are two dim3 functions for each kernel
                    {
                        errs()<<"CHICHI\n";
                        //Instruction * stream_def_inst = dyn_cast<Instruction>(stream_v);
                        //QUES.: In loop, it will skip the phi node, but we dont want to carry the 
                        //Find the def inst of Dim3_Inst's 2nd arg
                        Value * search_v = Dim3_Inst->getArgOperand(1);
                        if(isa<ConstantInt>(search_v))
                        {
                            errs()<<"First!\n";
                            addBB(direct_pred_bb);
                            one_bb_split_flag = true;
                            continue;
                        }
                        Instruction * v_def_inst = dyn_cast<Instruction>(search_v);
                        errs()<<""<<*v_def_inst<<"\n";
                        if(v_def_inst == dyn_cast<Instruction>(direct_pred_bb->getFirstInsertionPt()))
                        {
                            errs()<<"First!\n";
                            addBB(direct_pred_bb);
                            one_bb_split_flag = true;
                            continue;
                        }
                        else
                        {
                            bbs_to_split.push_back(direct_pred_bb);
                            splited_node_insts.push_back(v_def_inst);
                            one_bb_split_flag = true;
                        }
                    }
                }
            }
        }
        //errs()<<"Begin to split\n";
        for(size_t i = 0; i < bbs_to_split.size(); i++)
        {
            BasicBlock * new_pred_bb = SplitBlock(bbs_to_split[i],splited_node_insts[i]);
            addBB(new_pred_bb);
        }
        addBB(cur_bb);
        //errs()<<"Finish CHICHI\n";
    }

    void GPUFuncNode::SetStream(GlobalVariable * stream, Module & M)
    {
        //errs()<<"Set stream for gpufunc\n";
        std::vector<BasicBlock*> own_bbs = this->getBBs();
        Instruction * PushCallConfig_Inst = nullptr;

        //Locate the hipPushCallConfig inst, which is the first inst of its BB
        for(BasicBlock * bb : own_bbs)
        {
            for(auto inst = bb->begin(), inst_end = bb->end(); inst != inst_end; inst++)
            {
                if(isa<CallInst>(inst))
                {
                    CallInst * call_inst = dyn_cast<CallInst>(inst);
                    Function * called_func = call_inst->getCalledFunction();
                    if(called_func == nullptr) continue;
                    else
                    {
                        if(called_func->getName().str() == "__cudaPushCallConfiguration")
                        {
                            //errs()<<"Find it\n";
                            PushCallConfig_Inst = call_inst;
                        }
                    }
                }
            }
        }
        if(PushCallConfig_Inst == nullptr)
        {
            errs()<<"Cannot find the PushCallConfig_Inst for \n";
            this->dump_inst();
            exit(1);
        }
        //insert the load inst of stream
        IRBuilder<> builder(PushCallConfig_Inst);
        Value * stream_v = dyn_cast<Value>(stream);
        //errs()<<"Stream: "<<*stream_v<<"\n";
        Value * cur_stream = builder.CreateLoad(stream_v,"");
        if(cur_stream == nullptr)
        {
            errs()<<"Cannot create stream load inst\n";
            exit(1);
        }
        //bitcast cur_stream to i8*
        PointerType * int8PtrType = PointerType::getUnqual(Type::getInt8Ty(M.getContext()));
        if(int8PtrType == nullptr)
        {
            errs()<<"cannot get the int8 ptr type\n";
            exit(1);
        }
        Value * Int8_Stream_v = builder.CreateBitCast(cur_stream,int8PtrType,"");
        
        //replace the stream arg in pushCallConfig_Inst
        PushCallConfig_Inst->setOperand(5,Int8_Stream_v);
    }

    CPUFuncNode::CPUFuncNode(CallInst * ci, Function * cf, bool gpu_f, bool graph_f):FuncNode(ci,cf,gpu_f,graph_f)
    {
        //do nothing?
    }

    MemcpyNode::MemcpyNode(CallInst * ci, Function * cf, bool gpu_f = false, bool graph_f = false):FuncNode(ci,cf,gpu_f,graph_f)
    {
        //Init the input/output of memcpy node
        Value * input_arg = ci->getArgOperand(1);
        Value * output_arg = ci->getArgOperand(0);

        //TO.DO.: Need to identify whether we need to bitcast pointer type to i8 *. If passed a i8*, then bitcast is unneccessary
        if(BitCastInst * input_bitcast_inst = dyn_cast<BitCastInst>(input_arg))
        {
            Value * mid_v = input_bitcast_inst->getOperand(0);
            if(LoadInst * ld_inst = dyn_cast<LoadInst>(mid_v))
            {
                Value * container_v = ld_inst->getOperand(0);
                bool flag = false;
                for(auto user = container_v->user_begin(), user_end = container_v->user_end();
                    user != user_end; user++)
                {
                    if(StoreInst * st_inst = dyn_cast<StoreInst>(*user))
                    {
                        Value * original_v = st_inst->getOperand(0);
                        add_input_value(original_v);
                        flag = true;
                        break;
                    }
                }
                if(!flag)
                {
                    errs()<<"Cannot find the original input arg of ";
                    errs()<<*ci<<"\n";
                    exit(1);
                }
            }
            else
            {
                errs()<<"Cannot get the ld_inst of input_arg in MemcpyInst ";
                errs()<<*ci<<"\n";
                exit(1);
            }
        }
        else
        {   //No bitcast, original value is i8*
            if(LoadInst * ld_inst = dyn_cast<LoadInst>(input_arg))
            {
                Value * container_v = ld_inst->getOperand(0);
                bool flag = false;
                for(auto user = container_v->user_begin(), user_end = container_v->user_end();
                    user != user_end; user++)
                {
                    if(StoreInst * st_inst = dyn_cast<StoreInst>(*user))
                    {
                        Value * original_v = st_inst->getOperand(0);
                        add_input_value(original_v);
                        flag = true;
                        break;
                    }
                }
                if(!flag)
                {
                    errs()<<"Cannot find the original input arg of ";
                    errs()<<*ci<<"\n";
                    exit(1);
                }
            }
        }
        if(BitCastInst * output_bitcast_inst = dyn_cast<BitCastInst>(output_arg))
        {
            Value * mid_v = output_bitcast_inst->getOperand(0);
            if(LoadInst * ld_inst = dyn_cast<LoadInst>(mid_v))
            {
                Value * container_v = ld_inst->getOperand(0);
                bool flag = false;
                for(auto user = container_v->user_begin(), user_end = container_v->user_end();
                    user != user_end; user++)
                {
                    if(StoreInst * st_inst = dyn_cast<StoreInst>(*user))
                    {
                        Value * original_v = st_inst->getOperand(0);
                        add_output_value(original_v);
                        flag = true;
                        break;
                    }
                }
                if(!flag)
                {
                    errs()<<"Cannot find the original output arg of ";
                    errs()<<*ci<<"\n";
                    exit(1);
                }
            }
            else
            {
                errs()<<"Cannot get the ld_inst of output_arg in MemcpyInst ";
                errs()<<*ci<<"\n";
                exit(1);
            }
        }
        else
        {   //No bitcast, original value is i8*
            if(LoadInst * ld_inst = dyn_cast<LoadInst>(output_arg))
            {
                Value * container_v = ld_inst->getOperand(0);
                bool flag = false;
                for(auto user = container_v->user_begin(), user_end = container_v->user_end();
                    user != user_end; user++)
                {
                    if(StoreInst * st_inst = dyn_cast<StoreInst>(*user))
                    {
                        Value * original_v = st_inst->getOperand(0);
                        add_output_value(original_v);
                        flag = true;
                        break;
                    }
                }
                if(!flag)
                {
                    errs()<<"Cannot find the original output arg of ";
                    errs()<<*ci<<"\n";
                    exit(1);
                }
            }
        }
    }

    void MemcpyNode::CollectBBs(Module & M)
    {
        //errs()<<"KAKA\n";
        CallInst * callmemcpy_inst = this->getCallInst();

        //Find the first instruction of own bb
        Instruction * head_inst = nullptr;
        Value * first_operand = callmemcpy_inst->getArgOperand(0);
        if(LoadInst * ld_inst = dyn_cast<LoadInst>(first_operand))
        {
            //its original type is i8*
            head_inst = ld_inst;
        }
        else
        {
            BitCastInst * cast_inst = dyn_cast<BitCastInst>(first_operand);
            Value * ld_v = cast_inst->getOperand(0);
            if(LoadInst * ld_inst = dyn_cast<LoadInst>(ld_v))
            {
                head_inst = ld_inst;
            }
            else
            {
                errs()<<"Wrong ld_inst\n";
                exit(1);
            }
        }
        errs()<<*head_inst<<"\n";
        BasicBlock * old_bb = callmemcpy_inst->getParent();
        BasicBlock * cur_bb = nullptr;
        if(head_inst == dyn_cast<Instruction>(old_bb->getFirstInsertionPt()))
            cur_bb = old_bb;
        else 
            cur_bb = SplitBlock(old_bb,head_inst);
        
        Instruction * partitioned_node = callmemcpy_inst->getNextNode();
        //NOTE: Or we can use getNextNonDebugInstruction() to skip debug instructions
        SplitBlock(cur_bb,partitioned_node);
        this->addBB(cur_bb);
        //errs()<<"Finish KAKA\n";
    }

    void MemcpyNode::SetStream(GlobalVariable * stream, Module & M)
    {
        //errs()<<"Set Stream for memcpy\n";
        //TO.DO.:need to replace it with memcpyAsync()
        CallInst * call_inst = this->getCallInst();
        size_t arg_num = call_inst->getNumArgOperands();
        std::vector<Type*> args_type;
        std::vector<Value*> args;
        for(size_t i = 0; i < arg_num; i++) 
        {
            Value * cur_v = call_inst->getArgOperand(i);
            args.push_back(cur_v);
            args_type.push_back(cur_v->getType());
        }

        IRBuilder<> builder(call_inst);
        Value * stream_v = builder.CreateLoad(dyn_cast<Value>(stream),"");
        if(stream_v == nullptr)
        {
            errs()<<"Cannot load stream for memcpy\n";
            exit(1);
        }
        args.push_back(stream_v);

        //Anounce the MemcpyAsync()
        auto cuStream_t = StructType::getTypeByName(M.getContext(), "struct.CUstream_st");
        auto cuStream_t_PtrT = PointerType::get(cuStream_t,0);
        args_type.push_back(cuStream_t_PtrT);
        FunctionType * i32_FuncType = FunctionType::get(Type::getInt32Ty(M.getContext()),args_type,false);
        FunctionType * MemcpyAsync_i32_FuncType = FunctionType::get(Type::getInt32Ty(M.getContext()),args_type,false);
        
        Function * MemcpyAsyncFunc = nullptr;
        for(auto func = M.getFunctionList().begin(); func != M.getFunctionList().end(); func++)
        {
            Function * cur_func = dyn_cast<Function>(func);
            if(cur_func->getName().str() == "cudaMemcpyAsync")
            {
                MemcpyAsyncFunc = cur_func;
            }
            else continue;
        }
        
        if(!MemcpyAsyncFunc)
        {
            MemcpyAsyncFunc = Function::Create(MemcpyAsync_i32_FuncType,Function::ExternalLinkage,"cudaMemcpyAsync",M);
        }

        Value * ret_v = builder.CreateCall(i32_FuncType,MemcpyAsyncFunc,makeArrayRef(args));
        if(!ret_v)
        {
            errs()<<"Cannot create call of MemcpyAsyncFunc()\n";
            exit(1);
        }
        call_inst->replaceAllUsesWith(ret_v);
        call_inst->eraseFromParent();
        CallInst * new_callinst = dyn_cast<CallInst>(ret_v);
        setCallInst(new_callinst);
        setCalledFunc(MemcpyAsyncFunc);
    }

    Seq_Graph::Seq_Graph():Node()
    {
        Entry_Node = new Node();
        End_Node = new Node();
        Entry_Node->clear_pred();
        Entry_Node->add_succ(End_Node);
        End_Node->add_pred(Entry_Node);
        End_Node->clear_succ();
    }

    Seq_Graph::~Seq_Graph()
    {
        delete Entry_Node;
        delete End_Node;
        Entry_Node = nullptr;
        End_Node = nullptr;
    }

    void Seq_Graph::Insert(Node * target_node, Node * pred_node)
    {
        Node * next_node = pred_node->get_succ(0);
        pred_node->clear_succ();
        pred_node->add_succ(target_node);
        target_node->add_pred(pred_node);
        target_node->add_succ(next_node);
        next_node->clear_pred();
        next_node->add_pred(target_node);
    }

    Node * Seq_Graph::get_last_Node()
    {
        Node * last_node = End_Node->get_pred(0);
        return last_node;
    }

    bool Seq_Graph::IsEndNode(Node * node)
    {
        return node == End_Node;
    }

    Node * Seq_Graph::getEntryNode()
    {
        return Entry_Node;
    }

    void Seq_Graph::WalkGraph()
    {
        //TO.DO.
    }

    void Seq_Graph::Print_allNode()
    {
        Node * cur_node = getEntryNode();
        cur_node = cur_node->get_succ(0);
        errs()<<"<<<<<<<<<<<<<<<<<<<<\n";
        while(!IsEndNode(cur_node))
        {
            if(cur_node == nullptr)
            {
                std::cout<<"Cannot get func_node of cur_node"<<std::endl;
                exit(1);
            }
            //cur_node->dump_func();
            cur_node->dump_inst();
            std::vector<Value*> input_v = cur_node->get_input_value();
            std::vector<Value*> output_v = cur_node->get_output_value();
            errs()<<"Input Value:\n";
            for(auto v : input_v)
            {
                errs()<<*v<<"\n";
            }
            errs()<<"Output Value:\n";
            for(auto v : output_v)
            {
                errs()<<*v<<"\n";
            }

            errs()<<"BasicBlock:\n";
            cur_node->dumpBBs();
            
            //cur_node->dump_inst();
            cur_node = cur_node->get_succ(0);
        }
        errs()<<"<<<<<<<<<<<<<<<<<<<<\n";
        return;
    }

    void Seq_Graph::CollectBBs(Module & M)
    {
        Node * cur_node = getEntryNode();
        cur_node = cur_node->get_succ(0);
        while(!IsEndNode(cur_node))
        {
            if(cur_node == nullptr)
            {
                std::cout<<"Cannot get func_node of cur_node"<<std::endl;
                exit(1);
            }
            cur_node->CollectBBs(M);
            cur_node = cur_node->get_succ(0);
        }
        return;
    }
    
    DAG::DAG():Node()
    {
        Entry_Node = new Node();
        End_Node = new Node();
        Entry_Node->clear_pred();
        End_Node->clear_succ();
        Entry_Node->add_succ(End_Node);
        End_Node->add_pred(Entry_Node);
    }

    DAG::~DAG()
    {
        delete Entry_Node;
        Entry_Node = nullptr;
        delete End_Node;
        End_Node = nullptr;
    }

    void DAG::ConstructFromSeq_Graph(Seq_Graph * SG)
    {
        std::vector<Node*> empty_vec;
        pred_map[Entry_Node] = empty_vec;
        pred_map[End_Node].push_back(Entry_Node);

        Node * Seq_EntryNode = SG->getEntryNode();
        Node * Seq_G_cur_node = Seq_EntryNode->get_succ(0);
        while(!SG->IsEndNode(Seq_G_cur_node))
        {
            this->Insert(Seq_G_cur_node);
            Seq_G_cur_node = Seq_G_cur_node->get_succ(0);
        }
    }

    void DAG::Insert(Node * node)                   //Here we get the func_node in Seq_Graph
    {
        //Get all needed info of node to init node in dag
        std::vector<Value *> input_v = node->get_input_value();
        std::vector<Value *> output_v = node->get_output_value();
        std::vector<BasicBlock *> bbs = node->getBBs();
        CallInst * call_inst = node->getCallInst();
        Function * called_func = node->getFunction();

        //errs()<<"Inserting ";
        //node->dump_inst();

        //QUES.: Should we specify it as categoried function?
        

        FuncNode * new_node;
        if(dynamic_cast<GPUFuncNode*>(node))
        {
            new_node = new GPUFuncNode(call_inst,called_func,true,true);
        }
        else if(dynamic_cast<MemcpyNode*>(node))
        {
            new_node = new MemcpyNode(call_inst,called_func,true,true);
        }
        else
        {
            errs()<<"Error func node\n";
            exit(1);
        }
        for(BasicBlock * bb : bbs) new_node->addBB(bb);


        //Here we should follow the WAW, WAR, RAW order, so we should find the last pred node using following rule:
        //1. compare new_node's input value to all possible pred's output
        //2. compare new_node's output value to all possible pred's input
        //3. compare new_node's output value to all possible pred's output
        std::vector<Node *> pred_nodes;

        //RAW
        for(size_t i = 0; i < input_v.size(); i++)
        {
            //Here we make all read and write value of current kernel function into its input_v
            Value * cur_v = input_v[i];
            Type * type = cur_v->getType();
            if(type->isPointerTy())
            {
                new_node->add_input_value(cur_v);
                
                // Node * pred_node = reverse_find_pred(cur_v,true,true)[0];
                //NOTE: To find RAW, we might find two pred, where Entry_Node lies. Select another one if the first one is Entry_Node
                std::vector<Node*> RAW_pred_nodes = reverse_find_pred(cur_v, true, true);
                Node * pred_node = RAW_pred_nodes[0];
                for(Node * RAW_pred_node : RAW_pred_nodes)
                {
                    if(RAW_pred_node != Entry_Node)
                    {
                        pred_node = RAW_pred_node;
                        break;
                    }
                }
                
                /*
                if(pred_node == Entry_Node) errs()<<"raw_Pre_Node is Entry Node\n";
                else pred_node->dump_inst();
                */
                
                if(pred_node==nullptr)
                {
                    errs()<<"Cannot find the raw_pred_node\n";
                    exit(1);
                }
                //NOTE: we may not only return one node, and we should avoid multiple nodes outputing the same value
                //in the same level. TO.DO.: When inserting node, we should follow WAW order.
                //We should check whether pred_node is right before End_Node, 
                //if so, the new_node will be inserted between them, End_Node should be cleared from succ of pred_node
                //if not, End_Node should not be cleared
                /*                
                bool Last_Node_Flag = false;
                size_t pred_node_succ_num = pred_node->get_succ_num();
                //Determine whether it is the last tail of current branch like entry->xxx->(new_node)->end_node
                for(size_t j = 0; j < pred_node_succ_num; j++)
                {
                    Node * pred_node_succ = pred_node->get_succ(j);
                    if(pred_node_succ == End_Node)
                    {
                        Last_Node_Flag = true;
                        break;
                    }
                }
                */
                //if(Last_Node_Flag)  pred_node->clear_succ(End_Node);
                //pred_node->add_succ(new_node);
                pred_nodes.push_back(pred_node);
            }
            else
                continue;
        }

        //DEBUG: In benchmarkb8, _Z29__device_stub__minimum_kerneliPfS_i cannot find its correct pred, test for it          DONE!
        // if(new_node->get_func_name() == "_Z29__device_stub__minimum_kerneliPfS_i")
        // {
        //     errs()<<"RAW preds found for _Z29__device_stub__minimum_kerneliPfS_i\n";
        //     for(auto pred_node : pred_nodes)
        //     {
        //         if(pred_node == Entry_Node) errs()<<"Entry Node\n";
        //         pred_node->dump_inst();
        //     }
        //     exit(1);
        // }

        //WAW & WAR
        for(size_t i = 0; i < output_v.size(); i++)
        {
            Value * cur_v = output_v[i];
            Type * type = cur_v->getType();
            if(type->isPointerTy())
            {
                new_node->add_output_value(cur_v);
                
                Node * waw_pred_node = reverse_find_pred(cur_v,true,true)[0];
                if(waw_pred_node==nullptr)
                {
                    errs()<<"Cannot find the waw_pred_node\n";
                    exit(1);
                }
                /*
                if(waw_pred_node == Entry_Node) errs()<<"waw_Pre_Node is Entry Node\n";
                else waw_pred_node->dump_inst();
                */
                std::vector<Node *> war_pred_nodes = reverse_find_pred(cur_v,false,false);

                /*
                errs()<<">>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
                node->dump_inst();
                errs()<<"WAR Pred Nodes:\n";
                for(auto war_pred_node : war_pred_nodes) war_pred_node->dump_inst();
                errs()<<">>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
                */

                if(war_pred_nodes.size()==0)
                {
                    errs()<<"Cannot find the war_pred_node\n";
                    exit(1);
                }
                /*
                if(war_pred_node == Entry_Node) errs()<<"war_Pre_Node is Entry Node\n";
                else war_pred_node->dump_inst();
                */
                /*
                bool waw_pred_node_Last_Node_Flag = false;
                bool war_pred_node_Last_Node_Flag = false;
                size_t waw_pred_node_succ_num = waw_pred_node->get_succ_num();
                size_t war_pred_node_succ_num = war_pred_node->get_succ_num();
                for(size_t j = 0; j < waw_pred_node_succ_num; j++)
                {
                    if(waw_pred_node->get_succ(j) == End_Node)
                    {
                        waw_pred_node_Last_Node_Flag = true;
                        break;
                    }
                }
                for(size_t j = 0; j < war_pred_node_succ_num; j++)
                {
                    if(war_pred_node->get_succ(j) == End_Node)
                    {
                        war_pred_node_Last_Node_Flag = true;
                        break;
                    }
                }
                */
                //if(war_pred_node_Last_Node_Flag) war_pred_node->clear_succ(End_Node);
                //if(waw_pred_node_Last_Node_Flag) waw_pred_node->clear_succ(End_Node);

                //war_pred_node->add_succ(new_node);
                //waw_pred_node->add_succ(new_node);
                for(auto war_pred_node : war_pred_nodes) pred_nodes.push_back(war_pred_node);
                pred_nodes.push_back(waw_pred_node);
            }
            else continue;
        }

        new_node->add_succ(End_Node);
        End_Node->add_pred(new_node);
        pred_map[End_Node].push_back(new_node);

        //Delete EntryNode from pred of End_Node
        /*
        bool entry_node_is_pred_of_end_node = false;
        size_t EndNode_pred_num = End_Node->get_pred_num();
        for(size_t i = 0; i < EndNode_pred_num; i++)
        {
            if(End_Node->get_pred(i) == Entry_Node)
            {
                entry_node_is_pred_of_end_node = true;
                break;
            }
        }
        if(entry_node_is_pred_of_end_node) End_Node->clear_pred(Entry_Node);
        */
        //Make them unique and remove EntryNode from them if they consist more than EntryNode
        std::unordered_map<Node*, bool> unique_pred_node_m;
        for(Node * pred_node : pred_nodes)
        {
            if(unique_pred_node_m.find(pred_node) != unique_pred_node_m.end()) continue;
            else unique_pred_node_m[pred_node] = true;
        }

        if(unique_pred_node_m.size() > 1 && unique_pred_node_m.find(Entry_Node) != unique_pred_node_m.end())
        {
            unique_pred_node_m.erase(Entry_Node);
        }

        for(auto it = unique_pred_node_m.begin(); it != unique_pred_node_m.end(); it++)
        {
            Node * pred_node = it->first;
            //clear EndNode out of pred_node's succ if they are adjunt before
            bool clear_end_node = false;
            size_t succ_num = pred_node->get_succ_num();
            for(size_t i = 0; i < succ_num; i++)
            {
                if(pred_node->get_succ(i) == End_Node)
                    clear_end_node = true;
            }
            if(clear_end_node) 
            {
                pred_node->clear_succ(End_Node);
                End_Node->clear_pred(pred_node);
                //delete pred_node from End_Node in pred_map
                size_t index = 0;
                for(Node * pred : pred_map[End_Node])
                {
                    if(pred == pred_node)
                        break;
                    else index++;
                }
                if(index < pred_map[End_Node].size())
                {
                    pred_map[End_Node].erase(pred_map[End_Node].begin() + index);
                }

            }

            pred_node->add_succ(new_node);
            new_node->add_pred(pred_node);
            pred_map[new_node].push_back(pred_node);
        }
        /*
        std::unordered_map<Node*,bool> m;
        for(Node * pred_node : pred_nodes)
        {
            if(m.find(pred_node) == m.end()) 
            {
                new_node->add_pred(pred_node);
                pred_map[new_node].push_back(pred_node);
                //clear EndNode out of pred_node's succ if they are adjunt before
                bool clear_end_node_flag = false;
                size_t succ_num = pred_node->get_succ_num();
                for(size_t i = 0; i < succ_num; i++)
                {
                    if(pred_node->get_succ(i) == End_Node)
                        clear_end_node_flag = true;
                }
                if(clear_end_node_flag) pred_node->clear_succ(End_Node);
                
                pred_node->add_succ(new_node);
                m[pred_node] = true;
            }
            else 
                continue;
        }
        */
    }

    void DAG::Delete(Node * node)
    {
        //TO.DO.
    }

    std::vector<Node*> DAG::reverse_find_pred(Value * v, bool find_pred_output, bool Only_flag)             //Only_flag is true means we are dealing WAW or RAW
                                                                                                            //because one value only can be modified once in one layer
                                                                                                            //and that's the latest pred
    {
        std::queue<Node*> q;
        q.push(End_Node);
        std::vector<Node*> ans;
        while(!q.empty())
        {
            int n = q.size();
            for(int i = 0; i < n; i++)
            {
                Node * cur_node = q.front();
                q.pop();
                if(cur_node == Entry_Node)
                {
                    ans.push_back(Entry_Node);
                }
                std::vector<Value*> target_values = find_pred_output ? cur_node->get_output_value() : cur_node->get_input_value();
                for(auto target_v : target_values)
                {
                    if(target_v == v)
                    {
                        ans.push_back(cur_node);
                        if(Only_flag) return ans;
                        else continue;
                    }
                }
                if(cur_node == Entry_Node) continue;
                size_t cur_pred_num = cur_node->get_pred_num();
                for(size_t j = 0; j < cur_pred_num; j++) 
                {
                    q.push(cur_node->get_pred(j));
                }
            }
        }
        
        return ans;
    }

    void DAG::dump()
    {
        //pred_map contains preds of nodes except Entry Node and End Node
        //Need to fullfill it
        
        
        errs()<<"Dumping pred_map\n";
        for(auto it = pred_map.begin(); it != pred_map.end(); it++)
        {
            errs()<<"Target node: ";
            if(it->first == Entry_Node) errs()<<"Entry Node\n";
            else if(it->first == End_Node) errs()<<"End Node\n";
            else it->first->dump_inst();

            for(auto pred : it->second)
            {
                errs()<<"Depends on ";
                if(pred == Entry_Node) errs()<<"Entry_Node\n";
                else pred->dump_inst();
            }
            errs()<<"\n";
        }

        std::unordered_map<Node *, std::vector<Node*>> rudu_map = pred_map;
        std::unordered_map<Node *, bool> vis_m;
        
        std::queue<Node*> q;
        q.push(Entry_Node);
        vis_m[Entry_Node] = true;
        //Only node with rudu = 0 can be in Queue
        int level = 0;
        while(!q.empty())
        {
            std::cout<<"********************\n";
            std::cout<<"Level "<<level++<<" :"<<std::endl;
            int n = q.size();
            if(n==1 && q.front() == End_Node)
            {
                errs()<<"End_Node\n";
                std::cout<<"********************\n";
                break;
            }
            for(int i = 0; i < n; i++)
            {
                Node * cur_node = q.front();
                if(cur_node != Entry_Node)
                    cur_node->dump_inst();
                else
                    errs()<<"Entry Node\n";
                
                q.pop();
                for(auto it = rudu_map.begin(), end = rudu_map.end(); it != end; it++)
                {
                    int target_index = 0;
                    int n_pred = it->second.size();
                    for(Node * pred : it->second)
                    {
                        if(pred == cur_node) break;
                        else target_index++;
                    }
                    if(target_index >= n_pred) continue;
                    else it->second.erase(it->second.begin() + target_index);
                }
                for(auto it = rudu_map.begin(), end = rudu_map.end(); it != end; it++)
                {
                    if(!vis_m[it->first] && it->second.size() == 0) 
                    {
                        q.push(it->first);
                        vis_m[it->first] = true;
                    }
                }
            }
            std::cout<<"********************\n";
        }
    }

    void DAG::levelize()
    {
        std::unordered_map<Node *, std::vector<Node*>> rudu_map = pred_map;
        std::unordered_map<Node *, bool> vis_m;
        
        std::queue<Node*> q;
        q.push(Entry_Node);
        vis_m[Entry_Node] = true;
        //Only node with rudu = 0 can be in Queue
        int level = 0;

        while(!q.empty())
        {
            //std::cout<<"********************\n";
            //std::cout<<"Level "<<level++<<" :"<<std::endl;
            int n = q.size();
            if(n==1 && q.front() == End_Node)
            {
                node_level_map[End_Node] = level;
                level_nodes_map[level].push_back(End_Node);
                break;
            }
            for(int i = 0; i < n; i++)
            {
                Node * cur_node = q.front();
                node_level_map[cur_node] = level;
                level_nodes_map[level].push_back(cur_node);
                q.pop();
                for(auto it = rudu_map.begin(), end = rudu_map.end(); it != end; it++)
                {
                    int target_index = 0;
                    int n_pred = it->second.size();
                    for(Node * pred : it->second)
                    {
                        if(pred == cur_node) break;
                        else target_index++;
                    }
                    if(target_index >= n_pred) continue;
                    else it->second.erase(it->second.begin() + target_index);
                }
                for(auto it = rudu_map.begin(), end = rudu_map.end(); it != end; it++)
                {
                    if(!vis_m[it->first] && it->second.size() == 0) 
                    {
                        q.push(it->first);
                        vis_m[it->first] = true;
                    }
                }
            }
            level++;
        }
        n_level = level-1;
    }

    void DAG::dump_level()
    {
        errs()<<"Dump Level Map\n";
        errs()<<"------------------\n";
        for(auto it = level_nodes_map.begin(); it != level_nodes_map.end(); it++)
        {
            errs()<<"Level "<<it->first<<" :\n";
            if(it->second.size() == 1 && it->second[0] == Entry_Node) errs()<<"Entry Node\n";
            else if(it->second.size() == 1 && it->second[0] == End_Node) errs()<<"End Node\n";
            else 
            {
                for(auto node : it->second)
                {
                    node->dump_inst();
                }
            }
        }
        errs()<<"----------------\n";
    }

    //Function to help sortPredByUndoneSucc
    bool cmp(std::pair<Node*,size_t> a, std::pair<Node*,size_t> b)
    {
        return a.second < b.second;
    }

    void DAG::StreamDistribute(StreamGraph * StreamG, GlobalVariable ** stream_var_ptrs, Module & M)
    {
        if(!n_level)                                                    //n_level means level num except Entry/End
        {
            errs()<<"Wrong Levelize DAG\n";
            exit(1);
        }
        
        size_t stream_num = StreamG->get_stream_num();
        
        srand(NULL);                                                    //To more randomly select a balanced stream

        for(size_t i = 1; i <= n_level; i++)
        {
            std::vector<Node*> nodes = level_nodes_map[i];
            std::unordered_map<Node*,bool> setted_flag_map;
            std::unordered_map<Node*,size_t> node_original_index_map;
            for(auto node: nodes) setted_flag_map[node] = false;
            
            //hash for those have no preds except EntryNode
            for(size_t node_index = 0; node_index < nodes.size(); node_index++)
            {
                Node * cur_node = nodes[node_index];
                node_original_index_map[cur_node] = node_index;
                if(cur_node->get_pred_num()==1 && cur_node->get_pred(0) == Entry_Node)
                {
                    size_t stream_id = node_index % stream_num;
                    StreamG->node_set_stream(cur_node,stream_id,stream_var_ptrs,M);
                    StreamG->init_node_undone_succ(cur_node);
                    //Entry_Node doesn't exist in StreamGraph, no need to reduce its undone succ
                    //Also, there is no EventEdge produced from this
                    setted_flag_map[cur_node] = true;
                }
            }

            //For the first level, work are done
            if(i == 1) continue;
            //Sort Nodes by their pred(from small to big)
            SortByPredNum(nodes,0,nodes.size()-1);

            std::vector<Node*> later_set_nodes;

            for(auto node : nodes)
            {
                if(setted_flag_map[node])
                {
                    errs()<<"Duplicated node to set stream\n";
                    exit(1);
                }

                int stream_end_pred = 0;             //when it's true, means that we need to find the optimal
                                                            //pred in those which are the end of streams
                std::vector<Node*> stream_end_preds;

                size_t pred_num = node->get_pred_num();
                std::vector<std::pair<Node*,size_t>> preds;
                //Sort pred by their undone succ(increase order)
                //Collect preds which are end of their streams
                for(size_t j = 0; j < pred_num; j++)
                {
                    Node * cur_pred = node->get_pred(j);
                    if(StreamG->node_is_end_of_stream(cur_pred)) 
                    {
                        stream_end_pred++;
                        stream_end_preds.push_back(cur_pred);
                    }
                    preds.push_back(std::pair<Node*,size_t>(cur_pred,StreamG->get_node_undone_succ(cur_pred)));
                }
                std::sort(preds.begin(), preds.end(), cmp);
                
                //at least one node will be undone succ for each pred(cur node)
                //It means no zero in preds' values
                
                if(!stream_end_pred)                //Situation 1. No optimal stream can be selected
                {
                    //QUES.: Should this be done together for all such nodes after others are set?
                    later_set_nodes.push_back(node);
                    continue;
                }
                else if(stream_end_pred == 1)       //Situation 2. only have one pred which is end of its stream
                {
                    
                    Node * target_pred = stream_end_preds[0];
                    size_t stream_id = StreamG->get_node_stream(target_pred);
                    StreamG->node_set_stream(node,stream_id,stream_var_ptrs,M);
                    StreamG->init_node_undone_succ(node);
                    //Add EventEdges
                    for(size_t j = 0; j < pred_num; j++)
                    {
                        Node * cur_pred = node->get_pred(j);
                        StreamG->reduce_node_undone_succ(cur_pred);                 //QUES.: Should the undone_succ_num only affect on the pred which is end of stream?
                        if(StreamG->get_node_stream(cur_pred) != stream_id)
                        {
                            StreamG->add_EE(cur_pred,node);
                        }
                    }
                    setted_flag_map[node] = true;
                }
                else                                //Situation 3. multiple quanlified preds which are end of their stream s
                {
                    //select the pred with least undone succ
                    for(auto it = preds.begin(), it_end = preds.end(); it != it_end; it++)
                    {
                        Node * target_pred = it->first;
                        std::vector<Node*>::iterator finder = std::find(stream_end_preds.begin(), stream_end_preds.end(), target_pred);
                        if(finder == stream_end_preds.end()) continue;
                        else
                        {
                            size_t stream_id = StreamG->get_node_stream(target_pred);
                            StreamG->node_set_stream(node,stream_id,stream_var_ptrs,M);
                            StreamG->init_node_undone_succ(node);
                            for(size_t j = 0; j < pred_num; j++)
                            {
                                Node * cur_pred = node->get_pred(j);
                                StreamG->reduce_node_undone_succ(cur_pred);
                                if(StreamG->get_node_stream(cur_pred) != stream_id) 
                                    StreamG->add_EE(cur_pred,node);
                            }
                            setted_flag_map[node] = true;
                            break;
                        }
                    }
                }

            }

            for(auto node : later_set_nodes)
            {
                //random pick one
                // size_t stream_id = node_original_index_map[node] % stream_num;
                size_t stream_id = rand() % stream_num;                                   
                StreamG->node_set_stream(node,stream_id,stream_var_ptrs,M);
                StreamG->init_node_undone_succ(node);
                size_t pred_num = node->get_pred_num();
                //Add EventEdges
                for(size_t j = 0; j < pred_num; j++)
                {
                    Node * cur_pred = node->get_pred(j);
                    StreamG->reduce_node_undone_succ(cur_pred);                 //QUES.: Should the undone_succ_num only affect on the pred which is end of stream?
                    if(StreamG->get_node_stream(cur_pred) != stream_id)
                    {
                        StreamG->add_EE(cur_pred,node);
                    }
                }
                setted_flag_map[node] = true;
            }
        }
    }

    int find_pivot(std::vector<Node*> & p, int s, int e)
    {
        size_t pivot = p[s]->get_pred_num();
        int bigger_index = e+1;
        for(int i = e; i > s; i--)
        {
            if(p[i]->get_pred_num() >= pivot)
            {
                bigger_index--;
                Node * tmp = p[i];
                p[i] = p[bigger_index];
                p[bigger_index] = tmp;
            }
        }
        bigger_index--;
        Node * tmp = p[s];
        p[s] = p[bigger_index];
        p[bigger_index] = tmp;
        return bigger_index;
    }

    void DAG::SortByPredNum(std::vector<Node*> & p, int s, int e)
    {
        if(s>=e) return;

        int pivot_index = find_pivot(p,s,e);
        SortByPredNum(p,s,pivot_index-1);
        SortByPredNum(p,pivot_index+1,e);
    }

    Node * DAG::get_EntryNode()
    {
        return Entry_Node;
    }

    EventEdge::EventEdge(Node * a, Node * b)
    {
        prev = a;
        succ = b;
    }

    void EventEdge::dump()
    {
        errs()<<"Prev: ";
        prev->dump_inst();
        errs()<<"Succ: ";
        succ->dump_inst();
    }

    StreamGraph::StreamGraph()
    {
        //do nothing
    }

    StreamGraph::StreamGraph(size_t n)
    {
        stream_n = n;
        for(size_t i = 0; i < n; i++)
        {
            stream_nodes_map[i] = std::vector<Node*>();
        }
    }

    StreamGraph::~StreamGraph()
    {
        //do nothing
    }

    void StreamGraph::add_EE(Node * pred, Node * cur)
    {
        size_t pred_stream_id = get_node_stream(pred);
        size_t pred_stream_index = get_node_stream_index(pred,pred_stream_id);
        std::vector<Node*> same_stream_preds;
        size_t pred_num = cur->get_pred_num();
        bool flag = true;                              //indicate whether we should add a event for current pred and cur
        for(size_t i = 0; i < pred_num; i++)
        {
            Node * cur_pred = cur->get_pred(i);
            size_t cur_pred_stream_id = get_node_stream(cur_pred);
            size_t cur_pred_stream_index = get_node_stream_index(cur_pred,cur_pred_stream_id);
            
            errs()<<"New add pred in position "<<pred_stream_index<<" of stream "<<pred_stream_id<<" is ";
            pred->dump_inst();
            errs()<<"Added pred in position "<<cur_pred_stream_index<<" of stream "<<cur_pred_stream_id<<" is ";
            cur_pred->dump_inst();
            errs()<<"cur: ";
            cur->dump_inst();
            
            if(cur_pred_stream_id != pred_stream_id) continue;
            else
            {
                if(pred_stream_index > cur_pred_stream_index)
                {
                    delete_EE(cur_pred,cur);
                }
                else if(pred_stream_index < cur_pred_stream_index)
                {
                    //delete_EE(pred,cur);
                    flag = false;
                }
            }
        }
        if(flag) 
        {
            
            errs()<<"Add Event:\n";
            errs()<<"Prev: ";
            pred->dump_inst();
            errs()<<"Tail: ";
            cur->dump_inst();
            
            EEs.push_back(EventEdge(pred,cur));
        }
    }

    void StreamGraph::delete_EE(Node * pred, Node * cur)
    {
        //delete the EE
        if(EEs.size() == 0) return;
        int target_index = -1;
        for(int i = 0; i < EEs.size(); i++)
        {
            if(EEs[i].prev == pred && EEs[i].succ == cur)
            {
                target_index = i;
                break;
            }
        }

        if(target_index == -1)
        {
            return;                                     //cannot find the target EE to delete
        }
        else
        {
            errs()<<"ddddddddd\n";
            pred->dump_inst();
            cur->dump_inst();
            errs()<<"ddddddddd\n";
            EEs.erase(EEs.begin() + target_index);
        }
    }

    void StreamGraph::fix_EEs()
    {
        //Following these rules:
        //1. A's event_pred can only be after B's event_pred when A is after B in a stream and their event_pred are in a stream
        //2. 
    }

    size_t StreamGraph::get_EE_num()
    {
        return EEs.size();
    }

    void StreamGraph::node_set_stream(Node * cur, size_t stream_id, GlobalVariable ** stream_var_ptrs, Module & M)
    {
        
        node_stream_map[cur] = stream_id;
        stream_nodes_map[stream_id].push_back(cur);
        cur->SetStream(stream_var_ptrs[stream_id], M);
        
    }

    size_t StreamGraph::get_node_stream(Node * cur)
    {
        if(node_stream_map.find(cur) == node_stream_map.end())
        {
            errs()<<"Not recorded in node_stream_map\n";
            exit(1);
        }
        else
        {
            return node_stream_map[cur];
        }
    }

    size_t StreamGraph::get_node_stream_index(Node * node, size_t stream_id)
    {
        size_t index = 0;
        for(Node * target_node : stream_nodes_map[stream_id])
        {
            if(target_node == node) break;
            else index++;
        }
        if(index == stream_nodes_map[stream_id].size()) 
        {
            errs()<<"Cannot get the index of node\n";
            exit(1);
        }
        return index;
    }

    bool StreamGraph::node_is_end_of_stream(Node * cur)
    {
        for(auto it = stream_nodes_map.begin(), it_end = stream_nodes_map.end(); it != it_end; it++)
        {
            if(it->second.empty()) continue;
            if(it->second.back() == cur) return true;
        }
        return false;
    }

    void StreamGraph::init_node_undone_succ(Node * node)
    {
        //We count EndNode in DAG here but it wont be the node we need to operate in StreamGraph
        size_t succ_num = node->get_succ_num();
        unset_succ_num_map[node] = succ_num;
    }

    void StreamGraph::reduce_node_undone_succ(Node * cur)
    {
        if(unset_succ_num_map.find(cur) == unset_succ_num_map.end())
        {
            errs()<<"Not recorded in unset_succ_num_map\n";
            exit(1);
        }
        else
        {
            unset_succ_num_map[cur]--;
        }
    }

    size_t StreamGraph::get_node_undone_succ(Node * cur)
    {
        if(unset_succ_num_map.find(cur) == unset_succ_num_map.end())
        {
            errs()<<"Not recorded in unset_succ_num_map\n";
            exit(1);
        }
        else
        {
            return unset_succ_num_map[cur];
        }
    }

    size_t StreamGraph::get_stream_num()
    {
        return stream_n;
    }    

    void StreamGraph::dump_Graph()
    {
        errs()<<"**********************\n";
        errs()<<"StreamGraph:\n";
        for(auto it = stream_nodes_map.begin(), it_end = stream_nodes_map.end(); it != it_end; it++)
        {
            size_t stream_id = it->first;
            errs()<<"The "<<stream_id<<"th Stream:\n";
            for(Node * node : it->second)
            {
                node->dump_inst();
            }
        }
        errs()<<"**********************\n";
    }

    void StreamGraph::dump_EEs()
    {
        errs()<<"**********************\n";
        errs()<<"EventEdge:\n";
        for(size_t i = 0; i < EEs.size(); i++)
        {
            errs()<<i+1<<"th Event:\n";
            EEs[i].dump();
        }
        errs()<<"**********************\n";

    }

    void StreamGraph::FuncCall_Reorder(BasicBlock * head_bb, BasicBlock * tail_bb, Node * EntryNode)
    {
        BasicBlock * last_bb = head_bb;
        std::unordered_map<Node*,bool> vis_m;
        //Use queue's top to ensure edge in one steam is satisfied or not
        std::vector<std::queue<Node*>> stream_queues(stream_n);

        for(int i = 0; i < stream_n; i++)
        {
            for(Node * node : stream_nodes_map[i])
            {
                stream_queues[i].push(node);
                vis_m[node] = false;
            }
        }
        vis_m[EntryNode] = true;

        bool flag = true;
        while(flag)
        {
            flag = false;
            for(int i = 0; i < stream_n; i++)
            {
                if(stream_queues[i].size() == 0) continue;
                Node * node = stream_queues[i].front();
                bool ready_flag = true;
                size_t pred_num = node->get_pred_num();
                for(size_t j = 0; j < pred_num; j++)
                {
                    Node * cur_pred = node->get_pred(j);
                    if(vis_m[cur_pred] == false) ready_flag = false;
                    else continue;
                }
                if(ready_flag)
                {
                    vis_m[node] = true;
                    ChangeLastBBSucc(last_bb,node,tail_bb,&last_bb);
                    flag = true;
                    stream_queues[i].pop();
                }
                else
                    continue;
            }
        }
        
        //Set the br of last node
        if(BranchInst * last_br_inst = dyn_cast<BranchInst>(last_bb->getTerminator()))
        {
            size_t operand_num = last_br_inst->getNumOperands();
            if(operand_num > 1)
            {
                errs()<<"Wrong last_br_inst\n";
                errs()<<*last_br_inst<<"\n";
                exit(1);
            }
            else
            {
                last_br_inst->setOperand(0,dyn_cast<Value>(tail_bb));
            }
        }
        else
        {
            errs()<<"Cannot get the br_inst of last bb\n";
            exit(1);
        }
        
    }


    void StreamGraph::ChangeLastBBSucc(BasicBlock * pred_bb, Node * node, BasicBlock * tail_bb, BasicBlock ** last_bb)
    {
        //Make the br of pred_bb jumps to node's first BB
        std::vector<BasicBlock*> node_bbs = node->getBBs();
        if(node_bbs.size() == 0)
        {
            errs()<<"This Node has no BBs\n";
            node->dump_inst();
            exit(1);
        }
        BasicBlock * target_head_bb = node_bbs[0];
        BasicBlock * target_tail_bb = node_bbs.back();
        Instruction * pred_terminator_inst = pred_bb->getTerminator();

        if(BranchInst * pred_br_inst = dyn_cast<BranchInst>(pred_terminator_inst))
        {
            size_t operand_num = pred_br_inst->getNumOperands();
            if(operand_num > 1)
            {
                errs()<<"Wrong pred_br_inst\n";
                errs()<<*pred_br_inst<<"\n";
                exit(1);
            }
            else
            {
                pred_br_inst->setOperand(0,dyn_cast<Value>(target_head_bb));
            }
        }
        else
        {
            errs()<<"Cannot get the branch inst of BB: ";
            pred_bb->printAsOperand(errs(),false);
            errs()<<"\n";
            exit(1);
        }

        Instruction * cur_terminator_inst = target_head_bb->getTerminator();
        if(BranchInst * cur_br_inst = dyn_cast<BranchInst>(cur_terminator_inst))
        {
            size_t operand_num = cur_br_inst->getNumOperands();
            if(operand_num > 1)
                cur_br_inst->setOperand(2,dyn_cast<Value>(tail_bb));            //br i1,label1,label2's operand index: 0,2,1
        }
        else
        {
            errs()<<"Cannot get the branch inst of BB: ";
            target_tail_bb->printAsOperand(errs(),false);
            errs()<<"\n";
            exit(1);
        }

        //update the last_bb to be the node's last BB
        BasicBlock * next_target_bb = node_bbs.back();
        *last_bb = next_target_bb;
    }

    void StreamGraph::create_Events(Module & M, BasicBlock * head_bb, GlobalVariable** stream_var_ptrs)
    {
        //Get the function ptr of EventCreate, EventRecord, StreamWaitEvent
        /*
        %117 = call i32 @cudaStreamWaitEvent(%struct.CUstream_st* %115, %struct.CUevent_st* %116, i32 0)
        %72 = call i32 @cudaEventRecord(%struct.CUevent_st* %70, %struct.CUstream_st* %71)
        %48 = call i32 @cudaEventCreate(%struct.CUevent_st** %22)
        */
        auto cuStream_t = StructType::getTypeByName(M.getContext(), "struct.CUstream_st");
        if(cuStream_t == nullptr)
        {
            cuStream_t = StructType::create(M.getContext(),"struct.CUstream_st");
        }
        auto cuStream_t_PtrT = PointerType::get(cuStream_t,0);
        
        auto cuEvent_t = StructType::getTypeByName(M.getContext(), "struct.CUevent_st");
        if(cuEvent_t == nullptr)
        {
            cuEvent_t = StructType::create(M.getContext(),"struct.CUevent_st");
        }
        auto cuEvent_t_PtrT = PointerType::get(cuEvent_t,0);
        auto cuEvent_t_PtrPtrT = PointerType::get(cuEvent_t_PtrT,0);
        
        auto i32_type = Type::getInt32Ty(M.getContext());

        std::vector<Type*> EventCreate_ArgTypes = {cuEvent_t_PtrPtrT};
        std::vector<Type*> EventRecord_ArgTypes = {cuEvent_t_PtrT,cuStream_t_PtrT};
        std::vector<Type*> StreamWaitEvent_ArgTypes = {cuStream_t_PtrT,cuEvent_t_PtrT,i32_type};

        FunctionType * EventCreate_FuncType = FunctionType::get(Type::getInt32Ty(M.getContext()),EventCreate_ArgTypes,false);
        FunctionType * EventRecord_FuncType = FunctionType::get(Type::getInt32Ty(M.getContext()),EventRecord_ArgTypes,false);
        FunctionType * StreamWaitEvent_FuncType = FunctionType::get(Type::getInt32Ty(M.getContext()),StreamWaitEvent_ArgTypes,false);

        if(!EventCreate_FuncType)
        {
            errs()<<"Cannot get EventCreateFuncType\n";
            exit(1);
        }
        else if(!EventRecord_FuncType)
        {
            errs()<<"Cannot get EventRecord_FuncType\n";
            exit(1);
        }
        else if(!StreamWaitEvent_FuncType)
        {
            errs()<<"Cannot get StreamWaitEvent_FuncType\n";
            exit(1);
        }

        Function * EventCreateFunc = nullptr;
        Function * EventRecordFunc = nullptr;
        Function * StreamWaitEventFunc = nullptr;
        //iterate through all function to find func_ptr for above functions
        for(auto func = M.getFunctionList().begin(); func != M.getFunctionList().end(); func++)
        {
            Function * cur_func = dyn_cast<Function>(func);
            if(cur_func->getName().str() == "cudaEventCreate")
            {
                EventCreateFunc = cur_func;
            }
            else if(cur_func->getName().str() == "cudaEventRecord")
            {
                EventRecordFunc = cur_func;
            }
            else if(cur_func->getName().str() == "cudaStreamWaitEvent")
            {
                StreamWaitEventFunc = cur_func;
            }
        }

        if(!EventCreateFunc)
            EventCreateFunc = Function::Create(EventCreate_FuncType,Function::ExternalLinkage,"cudaEventCreate",M);
        if(!EventRecordFunc)
            EventRecordFunc = Function::Create(EventRecord_FuncType,Function::ExternalLinkage,"cudaEventRecord",M);
        if(!StreamWaitEventFunc)
            StreamWaitEventFunc = Function::Create(StreamWaitEvent_FuncType,Function::ExternalLinkage,"cudaStreamWaitEvent",M);

        //Create event in head_bb and map each of them to specific node-pair
        Instruction * head_end_inst = head_bb->getTerminator();
        IRBuilder<> builder(head_end_inst);
        std::unordered_map<Value*,EventEdge*> m;

        for(size_t i = 0; i < EEs.size(); i++)
        {
            Value * event_value = builder.CreateAlloca(cuEvent_t_PtrT);
            if(!event_value)
            {
                errs()<<"Cannot create a event variable\n";
                exit(1);
            }
        
            m[event_value] = &EEs[i];

            std::vector<Value *> createEvent_args = {event_value};
            Value * ret_v = builder.CreateCall(EventCreateFunc,makeArrayRef(createEvent_args));
            if(ret_v == nullptr)
            {
                errs()<<"Cannot create eventCreate Func\n";
                exit(1);
            }
        
            Node * Prev = EEs[i].prev;
            Node * Succ = EEs[i].succ;

            //Get first instruction of first bb of succ_node, last instruction of last bb of prev_node
            Instruction * first_inst_Succ = dyn_cast<Instruction>(Succ->getBBs()[0]->getFirstInsertionPt());
            std::vector<BasicBlock *> Prev_BBs = Prev->getBBs();
            BasicBlock * last_bb_Prev = Prev_BBs.back();
            Instruction * last_inst_Prev = last_bb_Prev->getTerminator();

            //Get stream of Prev and Succ Node
            size_t prev_stream_index = get_node_stream(Prev);
            GlobalVariable * prev_stream_var = stream_var_ptrs[prev_stream_index];
            size_t succ_stream_index = get_node_stream(Succ);
            GlobalVariable * succ_stream_var = stream_var_ptrs[succ_stream_index];

            std::vector<Value*> args;

            //Insert cudaEventRecord() before last_inst_Prev
            IRBuilder<> builder(last_inst_Prev);
            ret_v = nullptr;
            
            //Get the arg of eventRecord
            ret_v = builder.CreateLoad(event_value);
            if(ret_v == nullptr)
            {
                errs()<<"Cannot load event value\n";
                exit(1);
            }
            args.push_back(ret_v);
            ret_v = nullptr;
            ret_v = builder.CreateLoad(prev_stream_var);
            if(ret_v == nullptr)
            {
                errs()<<"Cannot load stream value\n";
                exit(1);
            }
            args.push_back(ret_v);
            ret_v = nullptr;

            ret_v = builder.CreateCall(EventRecordFunc,makeArrayRef(args));
            if(ret_v == nullptr)
            {
                errs()<<"Cannot create call of eventRecord\n";
                exit(1);
            }
            ret_v = nullptr;

            while(!args.empty()) args.pop_back();

            //Insert cudaStreamWaitEvent() before first_inst_Succ          NOTE: The stream here should be the one of succ
            builder.SetInsertPoint(first_inst_Succ);
            ret_v = builder.CreateLoad(succ_stream_var);
            if(ret_v == nullptr)
            {
                errs()<<"Cannot load stream value\n";
                exit(1);
            }
            args.push_back(ret_v);
            ret_v = nullptr;
            ret_v = builder.CreateLoad(event_value);
            if(ret_v == nullptr)
            {
                errs()<<"Cannot load event value\n";
                exit(1);
            }
            args.push_back(ret_v);
            ret_v = nullptr;

            ConstantInt * constantInt_value = builder.getInt32(0);
            if(constantInt_value == nullptr)
            {
                errs()<<"Cannot get constant value of gemm id\n";
                exit(1);
            }
            Value * argv_int = dynamic_cast<Value*>(constantInt_value);
            args.push_back(argv_int);
            

            ret_v = builder.CreateCall(StreamWaitEventFunc,makeArrayRef(args));
            if(ret_v == nullptr)
            {
                errs()<<"Cannot create call of StreamWaitEventFunc\n";
                exit(1);
            }
        }



    }

}