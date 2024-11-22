import { Component, QueryList, ViewChildren, OnInit, Inject } from '@angular/core';
import { DataService } from '../data.service';
import { NgxSpinnerService } from 'ngx-spinner';
import { HttpClient, HttpHeaders } from '@angular/common/http';

import { PdfViewerComponent } from '../PdfViewer/pdf-viewer/pdf-viewer.component';
import { MAT_DIALOG_DATA, MatDialogRef,MatDialog } from '@angular/material/dialog';


@Component({
  selector: 'app-doc-match',
  templateUrl: './doc-match.component.html',
  styleUrls: ['./doc-match.component.css']
})


export class DocMatchComponent implements OnInit {

  docdata:any;
  loader:boolean = false
  allDepartments:any;
  selectFieldContainer:boolean = false
  overlayVisibility:boolean = false
  similarityObj: any;
  uncommonSection:any;
  similarity_keys: any;
  Uncommon_keys:any;
  correctResponse : boolean = false
  spinner:boolean = false;
  inputPath: any;
  sidebarVisible: boolean = false;
  company:any
  segregatedObj :any
  companyPairs:any
  department:any
  responsespinner=false
  data2 :any= { main_folder:"./sdf", companies: [] , department: '', threshold: 0.50}

  constructor(private dataService: DataService,
     private http: HttpClient,
      private loaderSvc: NgxSpinnerService,
      private dialog: MatDialog
      ) 
      {
  
      }

  toggleOverlay(similarity: any,uncommon:any) {
      this.similarityObj = similarity
      this.similarity_keys = Object.keys(this.similarityObj)
      this.overlayVisibility = true
      this.uncommonSection = uncommon
      this.Uncommon_keys = ["0"]
  }


  ngOnInit(): void {
  }





  
  sendData() {
 this.loader = true
    this.selectFieldContainer = false;
    const data = {
      main_folder: this.inputPath
    };

    this.dataService.categorizeFolder(JSON.stringify(data)).subscribe(
      response => {
       if(response){
        this.allDepartments = []
        const obj = JSON.parse(response.destination_structure)
        let companyArray = Object.keys(obj)
        // Create an array of objects
          this.company = companyArray.map((item, index) => {
            return {
                name: item,  
                id: index    
          }});

          companyArray.forEach(element => {
            const keys= Object.keys(obj[element])

            keys.forEach(key => {
              const newDepartment = {
                name: key,
                id: element 
            };

            this.allDepartments.push(newDepartment)
        })
            
            
          });

          this.selectFieldContainer = true;
          this.loader = false
       }
      },
      error => {
        console.error('Error:', error);
        this.loader = false
      }
    );
  }

  
parsethreshold(event:any){
  const threshold = event.target.value;
  this.data2.threshold = parseFloat(threshold)
}
viewPdf(file1:string,file2:string){
  const data = {
    pdf1_path:file1,
    pdf2_path:file2
  }
  this.loaderSvc.show()
  this.spinner= true;
  this.dataService.DocumentDetails(data).subscribe(
    response => {
      this.spinner= false;
      if(response){
        
        this.dialog.open(PdfViewerComponent,
          {
            maxWidth: '100vw',
            maxHeight: '100vh',
            height: '100%',
            width: '100%',
            panelClass: 'full-screen-modal',
            data:{
              pdfsrc1:file1,
              pdfsrc2:file2,
              response:response,
              path:this.inputPath,
              companypair:this.data2.companies,
              department:this.data2.department
            }
          })
      }
    },
    error => {
      this.spinner = false
      console.error('Error:', error);
      this.loaderSvc.hide()
    });
}

  sendData2(){
    this.loaderSvc.show()

    this.correctResponse = false
    this.spinner = true
    if(this.data2.companies.length == 1){
      const data = JSON.stringify(this.data2)
      this.dataService.inFolderWise(data).subscribe(
        response => {
              this.docdata = response.doc_details 
              this.segregatedObj = response.results
              this.companyPairs = Object.keys(this.segregatedObj)
              this.spinner = false
              this.loaderSvc.hide()
              this.correctResponse = true
            },
            error => {
              this.spinner = false
              console.error('Error:', error);
              this.loaderSvc.hide()
            }
          );

    }
    else{
      const data = JSON.stringify(this.data2)
      this.dataService.outFolder(data).subscribe(
        response => {
            this.segregatedObj = response.comparisons.reduce((acc: any, obj: any) => {
                const key = obj.company1;
                if (!acc[key]) {
                      acc[key] = []; // Create an array for this category if it doesn't exist
                  }
                  acc[key].push(obj); // Push the object into the corresponding category
                  return acc;
              }, {});
                this.companyPairs = Object.keys(this.segregatedObj)
                this.correctResponse = true
                this.spinner = false
                this.loaderSvc.hide()
  
        },
        error => {
          this.spinner = false
          console.error('Error:', error);
          this.loaderSvc.hide()
        }
      );
    }


      
  }


  onCompanySelection(nameArray: any){
    const departmentMap: { [key: string]: Set<string> } = {};

    // Populate the department map
    this.allDepartments.forEach((obj: any) => {
        if (!departmentMap[obj.name]) {
            departmentMap[obj.name] = new Set<string>();
        }
        departmentMap[obj.name].add(obj.id);
    });
    
    // Step 2: Filter departments that have both companies
    this.department = Object.keys(departmentMap)
        .filter(department => 
          nameArray.value.every((name: string) => departmentMap[department].has(name))
        );
    
  }
}
