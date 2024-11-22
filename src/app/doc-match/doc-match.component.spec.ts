import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DocMatchComponent } from './doc-match.component';

describe('DocMatchComponent', () => {
  let component: DocMatchComponent;
  let fixture: ComponentFixture<DocMatchComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ DocMatchComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(DocMatchComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
