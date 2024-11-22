import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { timeout } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class DataService {
  private apihost ='http://127.0.0.1:1010/'
  private apiUrl = 'http://192.168.1.10:9429/categorize/';
  private apiUrl2 = 'http://192.168.1.10:9429/out-folder-compare/'
  private apiUrl3 = 'http://192.168.1.10:9429/in-folder-compare/'

  constructor(private http: HttpClient) { }

  categorizeFolder(data:any): Observable<any> {
    const headers = new HttpHeaders({
      'Content-Type': 'application/json'
    });

    return this.http.post(this.apihost.concat('categorize/'), data, { headers });
  }

  DocumentDetails(data:any): Observable<any> {
    const headers = new HttpHeaders({
      'Content-Type': 'application/json'
    });

    return this.http.post(this.apihost.concat('pdf_overlay/'), data, { headers })// Increase timeout to 30 seconds or more
    ;
  }
  outFolder(data:any): Observable<any> {
    const headers = new HttpHeaders({
      'Content-Type': 'application/json'
    });

    return this.http.post(this.apihost.concat('out-folder-compare/'), data, { headers });
  }


  inFolderWise(data:any): Observable<any> {
    const headers = new HttpHeaders({
      'Content-Type': 'application/json'
    });

    return this.http.post(this.apihost.concat('in-folder-compare/'), data, { headers });
  }
  
}