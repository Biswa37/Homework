void convert_data_to_root(){
	TTree *tree = new TTree("data","data");
	tree->ReadFile("data1","E1:Px1:Py1:Pz1:E2:Px2:Py2:Pz2");
	tree->SaveAs("data.root");
}