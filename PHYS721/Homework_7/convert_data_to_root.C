void convert_data_to_root(){
	TTree *tree = new TTree("data","data");
	tree->ReadFile("data2","E");
	tree->SaveAs("data.root");
}