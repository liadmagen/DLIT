import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { FullyConnectedNnComponent } from './fully-connected-nn.component';

describe('FullyConnectedNnComponent', () => {
  let component: FullyConnectedNnComponent;
  let fixture: ComponentFixture<FullyConnectedNnComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ FullyConnectedNnComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(FullyConnectedNnComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
